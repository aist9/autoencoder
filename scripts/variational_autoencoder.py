
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

# Variational AutoEncoder class
class VariationalAutoEncoder(Chain):
    def __init__(self, layers, act_func='sigmoid'):
        # super(VariationalAutoEncoder, self).__init__()

        # 活性化関数の定義
        if act_func == 'sigmoid':
            self.act_func = F.sigmoid
        elif act_func == 'tanh':
            self.act_func = F.tanh
        elif act_func == 'relu':
            self.act_func = F.relu

        self.make_layers(layers)
  
    # callでは再構成のみを計算する
    def __call__(self, x, sigmoid=True):
        mu, var = self.encode(x)
        reconst = self.decode(z, sigmoid=sigmoid)
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
            l = L.Linear(layers[i], layers[i+1])
            name  = 'enc{}'.format(i)
            self.add_link(name, l)

            # デコード層をボトルネックから出力層の順で作成
            j = encoder_depth - i - 1
            l = L.Linear(layers[j], layers[j-1])
            name  = 'dec{}'.format(i)
            self.add_link(name, l)

        # muとsigmaを出力する層
        # この2つは分岐して作成される
        self.add_link('enc_mu' , L.Linear(layers[-2], layers[-1]))
        self.add_link('enc_var', L.Linear(layers[-2], layers[-1]))

        # 出力層
        self.add_link('dec_out', L.Linear(layers[1], layers[0]))
 
    # 特徴量が必要な場合はfeat_returm=True
    def encoder(self, x, feat_return=False):
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

    # lossにベルヌーイを使用する場合sigmoid=False
    def decoder(self, mu, var, sigmoid=True):
        reconst = z
        for layer in self.children():
            # デコード層のみ処理
            if 'out' in layer.name: 
                reconst = layer(reconst)
                if sigmoid:
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
            self.sigmoid = False

    # trainerで呼ばれるcall関数
    def __call__(self, x, t):
        # データ数
        data_num = x.shape[0]

        # Forwardとlossの計算
        mu, var = vae.encoder(x)
        for i in range(k):
            # MSEで誤差計算を行う
            if self.loss_function == 'mse':
                loss += F.mean_squared_error(x, reconst) / k

            # その他の場合はベルヌーイ分布により計算
            else:
                # bernoulli_nllがsigmoidを内包しているので学習時はsigmoid = False
                reconst = vae.decode(z, sigmoid=False)
                loss += F.bernoulli_nll(x, reconst) / (k * data_num)

        kld = gaussian_kl_divergence(mu, var) / data_num
        loss = loss + self.beta * kld 

        # Chainerのreport機能
        reporter.report({'loss': loss}, self)

        return loss


# 再構成と再構成誤差の計算
class Reconst():
    # 学習、モデルを渡しておく
    def __init__(self, model, sigmoid=False):
        
        self.model = model
        self.sig = sigmoid
    
    # 再構成と再構成誤差一括で計算
    def __call__(self, data):

        # 配列の次元数を調べる(1次元だとエラーを出すため)
        if data.ndim == 1:
            data = data.reshape(1, len(data))

        feat, reconst = self.data2reconst(data)
        err = self.reconst_err(data, reconst)
        return feat, reconst, err

    # 入力データを再構成
    def data2reconst(self, data):
        feat = Variable(data)
        mu,var = self.model.encode(feat)
        feat = F.gaussian(mu, var)
        
        reconst = feat
        reconst = self.model.decode(reconst,sigmoid=self.sig)

        return feat.data, reconst.data

    def decode(self,data):
        reconst = Variable(data)
        reconst = self.model.decode(reconst,sigmoid=self.sig)
        # reconst = self.model.decode(reconst,sigmoid=True)

        return reconst

    # 再構成誤差の計算
    def reconst_err(self, data, reconst):
        err = np.sum((data - reconst) ** 2, axis = 1) / data.shape[1]
        return err

    # 再構成誤差から平均と標準偏差を算出してしきい値を決める
    def err2threshold(self, err, sigma = 3):
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
    vae = VariationalAutoencoder(layers, act_func=act_func)
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
    trainer.extend(extensions.PrintReport( ['epoch', 'main/loss']))
    trainer.run()

    # GPUを使っていた場合CPUに戻す
    if -1 < gpu_device:
        vae.to_cpu()

    return vae


#
# def train_vae(model,train,epoch,batch,C=1.0,k=1,use_sigmoid=False):
#
#     model.to_gpu(0)
#     opt = optimizers.Adam()
#     opt.setup(model)
#
#     # あらかじめgpuに投げつつVariable化する
#     nlen = train.shape[0]
#     data = Variable(cuda.to_gpu(train))
#
#     # いつもの学習ループ
#     for ep in range(epoch):
#         perm = np.random.permutation(nlen)
#         for p in range(0,nlen,batch):
#             d = data[perm[p:p+batch]]
#
#             # エンコード
#             enc = d
#             model.cleargrads()
#             mu, ln_var = model.encode(enc)
#
#             rec_loss = 0
#             for l in range(k): #普通は k=1
#                 z = F.gaussian(mu, ln_var)
#
#                 # デコード
#                 dec = z
#                 dec = model.decode(dec, sigmoid=use_sigmoid)
#
#                 rec_loss += F.bernoulli_nll(d,dec) / (k * batch)
#                 # rec_loss += F.mean_squared_error(d,dec) / (k)
#
#             latent_loss = C * gaussian_kl_divergence(mu, ln_var) / batch
#             loss = rec_loss + latent_loss
#
#             # 逆伝搬
#             loss.backward()
#             opt.update()
#
#         if (ep + 1) % 10 == 0:
#             # print('\repoch ' + str(ep + 1) + ': ' + str(loss.data) + ' ', end = '')
#             print('\repoch ' + str(ep + 1) + ': ' + str(loss.data) + ' ',latent_loss,rec_loss ,end = '\n')
#             # print(model.a_enc0.W[0,0],model.a_enc1.W[0,0],model.a_enc2.W[0,0])
#             # print(model.b_dec0.W[0,0],model.b_dec1.W[0,0],model.b_dec2.W[0,0])
#     print()
#
#     model.to_cpu()
#
#     return model
#
# def train_stacked(train, hidden, epoch, batchsize, folder, \
#                   train_mode=True, \
#                   act=F.tanh,C=1.0,k=1, use_sigmoid=False ):
# 
#     inputs = train.shape[1]
#     if type(hidden) != type([]):
#         hidden = [hidden]
#     layer  = [inputs] + hidden
#     # layer.extend(hidden)
#    
#     # 隠れ層の数値を文字列にして保存・読み込みのフォルダを分ける
#     hidden_str = []
#     for i in range(len(hidden)):
#         hidden_str.append(str(int(hidden[i])))
#     hidden_num_str = '-'.join(hidden_str)
#     print('layer' + str(layer))
#
#     # 学習モデルの保存場所
#     folder_model = os.path.join(folder, hidden_num_str)
#     os.makedirs(folder_model, exist_ok=True)
#
#
#     # 隠れ層分だけloop
#     model = []
#     feat = train.copy()
#        
#     # 保存に使う文字列
#     hidden_num = ''
#     for i in layer:
#         hidden_num += str(i) + '_'
#     # モデルの保存名
#     save_name = os.path.join(folder_model, 'model_' + hidden_num + '.npz')
#     if train_mode:
#         model = VAE(layer, act)
#         model = train_vae(model, train, epoch, batchsize,C=C,k=k,use_sigmoid=use_sigmoid)
#         chainer.serializers.save_npz(save_name, model)
#
#     # 学習しない場合はloadする
#     else:
#         model = VAE(layer, act)
#         chainer.serializers.load_npz(save_name, model)
#
#     return model
#
