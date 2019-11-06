
import os, sys
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.computational_graph as c
from chainer import Chain
from chainer import Variable, optimizers
from chainer import datasets, iterators
from chainer import training
from chainer import reporter
from chainer.training import extensions

import cupy
import chainer.computational_graph as c
from chainer import Variable, optimizers, initializer,cuda
from chainer.functions.loss.vae import gaussian_kl_divergence

class Xavier(initializer.Initializer):
    """
    Xavier initializaer 
    Reference: 
    * https://jmetzen.github.io/2015-11-27/vae.html
    * https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    """
    def __init__(self, fan_in, fan_out, constant=1, dtype=None):
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.high = constant*np.sqrt(6.0/(fan_in + fan_out))
        self.low = -self.high
        super(Xavier, self).__init__(dtype)

    def __call__(self, array):
        xp = cuda.get_array_module(array)
        args = {'low': self.low, 'high': self.high, 'size': array.shape}
        if xp is not np:
            # Only CuPy supports dtype option
            if self.dtype == np.float32 or self.dtype == np.float16:
                # float16 is not supported in cuRAND
                args['dtype'] = np.float32
        array[...] = xp.random.uniform(**args)


# Variational AutoEncoder class
class VariationalAutoEncoder(Chain):
    def __init__(self, layers, act_func='sigmoid'):
        super(VariationalAutoEncoder, self).__init__()

        # 活性化関数の定義
        if act_func == 'sigmoid':
            self.act_func = F.sigmoid
        elif act_func == 'tanh':
            self.act_func = F.tanh
        elif act_func == 'relu':
            self.act_func = F.relu
        else:
            self.act_func = F.sigmoid


        self.make_layers(layers)
  
    # callでは再構成のみを計算する
    def __call__(self, x, use_sigmoid=True):
        mu, var = self.encoder(x)
        z = F.gaussian(mu, var)
        reconst = self.decoder(z, use_sigmoid=use_sigmoid)
        return reconst

    # レイヤー作成
    def make_layers(self, layers):
        # encoderの深さ
        encoder_depth = len(layers)

        # children()を使用するためnameは昇順にする
        # encoder: 入力層->ボトルネックへ昇順
        # decoder: ボトルネック->出力層へ昇順
        for i in range(encoder_depth - 2):
            # エンコード層を入力層からボトルネックの順で作成
            l = L.Linear(layers[i], layers[i+1], initialW=Xavier(layers[i], layers[i+1]))
            name  = 'enc{}'.format(i)
            self.add_link(name, l)

            # デコード層をボトルネックから出力層の順で作成
            j = encoder_depth - i - 1
            l = L.Linear(layers[j], layers[j-1], initialW=Xavier(layers[i], layers[i+1]))
            name  = 'dec{}'.format(i)
            self.add_link(name, l)

        # muとsigmaを出力する層
        # この2つは分岐して作成される
        self.add_link('enc_mu' , L.Linear(layers[-2], layers[-1], initialW=Xavier(layers[-2], layers[-1])))
        self.add_link('enc_var', L.Linear(layers[-2], layers[-1], initialW=Xavier(layers[-2], layers[-1])))

        # 出力層
        self.add_link('dec_out', L.Linear(layers[1], layers[0], initialW=Xavier(layers[1], layers[0])))
 
    # 特徴量が必要な場合はfeat_returm=True
    def encoder(self, x):
        feat = x
        for layer in self.children():
            # encoderのみ処理
            if 'mu' in layer.name:
                mu = layer(feat)
            elif 'var' in layer.name:
                var = layer(feat)
                break
            elif 'enc' in layer.name:
                feat = self.act_func(layer(feat))

        return mu, var

    # lossがベルヌーイ分布の場合のみuse_sigmoidで分岐させる
    def decoder(self, z, use_sigmoid=True):
        reconst = z
        for layer in self.children():
            # デコード層のみ処理
            if 'out' in layer.name: 
                reconst = layer(reconst)
                if use_sigmoid:
                    reconst = F.sigmoid(reconst)
                break
            if 'dec' in layer.name:
                reconst = self.act_func(layer(reconst))

        return reconst

# VAEをtrainerで学習するためのラッパー
class VariationalAutoencoderTrainer(Chain):
    def __init__(self, vae, beta=1.0, k=1, loss_function='mse'):
        super(VariationalAutoencoderTrainer, self).__init__(vae=vae)

        # 再構成誤差に用いる関数
        self.loss_function = loss_function
        if loss_function == 'bernoulli':
            self.use_act_out = False
            self.use_sigmoid = False
    
        self.k = k
        self.beta = beta

    # trainerで呼ばれるcall関数
    def __call__(self, x, t):
        # データ数
        num_data = x.shape[0]

        # Forwardとlossの計算
        mu, var = self.vae.encoder(x)
        z = F.gaussian(mu, var)
        reconst_loss = 0
        for i in range(self.k):
            # MSEで誤差計算を行う
            if self.loss_function == 'mse':
                reconst = self.vae.decoder(z, use_sigmoid=True)
                reconst_loss += F.mean_squared_error(x, reconst) / self.k

            # その他の場合はベルヌーイ分布により計算
            else:
                # bernoulli_nllがsigmoidを内包しているので学習時はsigmoid=False
                reconst = self.vae.decoder(z, use_sigmoid=False)
                reconst_loss += F.bernoulli_nll(x, reconst) / (self.k * num_data)

        kld = F.mean(gaussian_kl_divergence(mu, var))
        loss = reconst_loss + self.beta * kld 

        # Chainerのreport機能
        reporter.report({'loss': loss}, self)
        reporter.report({'reconst_loss': reconst_loss}, self)
        reporter.report({'kld': kld}, self)

        return loss


# 再構成と再構成誤差の計算
class Reconst():
    # 学習、モデルを渡しておく
    def __init__(self, model):
        
        self.model = model
    
    # 再構成と再構成誤差一括で計算
    def __call__(self, data):

        # 配列の次元数を調べる(1次元だとエラーを出すため)
        if data.ndim == 1:
            data = data.reshape(1, len(data))

        mu, var = self.model.encoder(data)
        z = F.gaussian(mu, var)
        reconst = self.model.decoder(z)
        err = self.reconst_err(data, reconst)

        return z.data, reconst.data, err

    # 再構成誤差の計算
    def reconst_err(self, data, reconst):
        err = np.sum((data - reconst.data) ** 2, axis = 1) / data.shape[1]
        return err

    # 再構成誤差から平均と標準偏差を算出してしきい値を決める
    def err_to_threshold(self, err, sigma = 3):
        mn  = np.mean(err)
        std = np.std(err)
        th = mn + sigma * std
        return mn, std, th
    
    # しきい値を超えた場合はFalse
    def judgment(self, err, th):
        result = [True if err[i] <= th else False for i in range(len(err))]
        return result


# trainerによるVAEの学習
def training_vae(data, hidden, max_epoch, batchsize, \
             act_func='sigmoid', gpu_device=0, \
             loss_function='mse'):
    
    # 入力サイズ
    inputs = data.shape[1]
    layers  = [inputs] + hidden

    # モデルの定義
    vae = VariationalAutoEncoder(layers, act_func=act_func)
    model = VariationalAutoencoderTrainer(vae, beta=1.0, k=1, loss_function=loss_function)
    opt = optimizers.Adam()
    opt.setup(model)

    # データの形式を変換する
    train = datasets.TupleDataset(data, data)
    train_iter = iterators.SerialIterator(train, batchsize)

    # 学習ループ
    updater = training.StandardUpdater(train_iter, opt, device=gpu_device)
    trainer = training.Trainer(updater, (max_epoch, 'epoch'), out="result")
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport( ['epoch', 'main/loss',
        'main/reconst_loss', 'main/kld', 'elapsed_time']))
    trainer.run()

    # GPUを使っていた場合CPUに戻す
    if -1 < gpu_device:
        vae.to_cpu()

    return vae

