
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.computational_graph as c
from chainer import Variable, optimizers, initializer,cuda
from chainer.functions.loss.vae import gaussian_kl_divergence

import cupy

import os

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


class VAE(chainer.Chain):
    """Variational AutoEncoder"""

    def __init__(self, inputs, hidden, act_func=F.tanh):
        super(VAE, self).__init__()
        self.act_func = act_func
        with self.init_scope():
            # encoder
            self.le        = L.Linear(inputs, hidden,      initialW=Xavier(inputs, hidden))
            self.le_var    = L.Linear(inputs, hidden,      initialW=Xavier(inputs, hidden))
            # self.le3_ln_var = L.Linear(n_h,  n_latent, initialW=Xavier(n_h,  n_latent))
            # decoder
            self.ld = L.Linear(hidden, inputs, initialW=Xavier(hidden, inputs))

    def __call__(self, x, sigmoid=True):
        """ AutoEncoder """
        e = self.encode(x)
        d = self.decode(e,sigmoid)
        return d


    def encode(self, x,isBottom=False):
        h1 = self.le(x)
        if isBottom:
            return h1, self.le_var(x)
        return self.act_func(h1)

    def decode(self, z,isTop=False, sigmoid=True):
        h1 = self.ld(z)
        if sigmoid and isTop:
            return F.sigmoid(h1)
        elif isTop:
            return h1
        else:
            return self.act_func(h1)



# 再構成と再構成誤差の計算
class Reconst():
    # 学習、モデルを渡しておく
    def __init__(self, model):
        
        # if type(model) != 'list':
            # model = [model]

        self.model = model
        self.L = len(model)
    
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
        for i in range(self.L-1):
            feat = self.model[i].encode(feat)
        mu,var = self.model[-1].encode(feat,isBottom=True)
        feat = F.gaussian(mu, var)
        
        reconst = feat
        for i in range(self.L-1):
            reconst = self.model[self.L - i - 1].decode(reconst)
        reconst = self.model[0].decode(reconst,isTop=True)

        return feat.data, reconst.data

    def decode(self,data):
        reconst = Variable(data)
        for i in range(self.L-1):
            reconst = self.model[self.L - i - 1].decode(reconst)
        reconst = self.model[0].decode(reconst,isTop=True)
        #reconst = self.model[0].decode(reconst,isTop=True,sigmoid=False)

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


def train_vae(models,train,epoch,batch,C=1.0,k=1):

    opts=[]
    for model in models:
        model.to_gpu(0)
        opt = optimizers.Adam()
        opt.setup(model)
        opts.append(opt)

    # あらかじめgpuに投げつつVariable化する
    nlen = train.shape[0]
    data = Variable(cuda.to_gpu(train))

    # いつもの学習ループ
    for ep in range(epoch):
        perm = np.random.permutation(nlen)
        for p in range(0,nlen,batch):
            d = data[perm[p:p+batch]]

            # エンコード
            enc = d
            for model in models[:-1]:
                model.cleargrads()
                enc = model.encode(enc)
            # ボトルネック層は2出力らしいので最後のエンコードのみ別処理
            models[-1].cleargrads()
            mu, ln_var = models[-1].encode(enc,isBottom=True)

            rec_loss = 0
            for l in range(k): #普通は k=1
                z = F.gaussian(mu, ln_var)

                # デコード
                dec = z
                for model in models[:0:-1]:
                    dec = model.decode(dec)
                #最終層は活性化なし
                #bernoulli_nllがsigmoidを内包しているため,sigmoid=False
                dec = models[0].decode(dec, isTop=True, sigmoid=False)
                rec_loss += F.bernoulli_nll(d,dec) / (k * batch)

            latent_loss = C * gaussian_kl_divergence(mu, ln_var) / batch
            loss = rec_loss + latent_loss

            # 逆伝搬
            loss.backward()
            # 各層は別々のモデルなので、各層にupdate処理。なんかいい方法ないかな
            for opt in opts:
                opt.update()

        if (ep + 1) % 10 == 0:
            #print('\repoch ' + str(ep + 1) + ': ' + str(loss.data) + ' ', end = '')
            print('\repoch ' + str(ep + 1) + ': ' + str(loss.data) + ' ', latent_loss,rec_loss,end = '\n')

    print()

    for model in models:
        model.to_cpu()

    return models

def train_stacked(train, hidden, epoch, batchsize, folder, \
                  train_mode=True, \
                  act=F.tanh):
 
    inputs = train.shape[1]
    layer  = [inputs] + hidden
    # layer.extend(hidden)
    
    # 隠れ層の数値を文字列にして保存・読み込みのフォルダを分ける
    hidden_str = []
    for i in range(len(hidden)):
        hidden_str.append(str(int(hidden[i])))
    hidden_num_str = '-'.join(hidden_str)
    print('layer' + str(layer))

    # 学習モデルの保存場所
    folder_model = os.path.join(folder, hidden_num_str)
    os.makedirs(folder_model, exist_ok=True)


    # 隠れ層分だけloop
    model = []
    feat = train.copy()
    for i,(l_i, l_o) in enumerate(zip(layer[0:-1], layer[1:])):
        
        # 保存に使う文字列
        hidden_num = str(l_i) + '_' + str(l_o)
        # モデルの保存名
        save_name = os.path.join(folder_model, 'model_' + hidden_num + '.npz')
        if train_mode:
            model.append(VAE(l_i, l_o, act))

        # 学習しない場合はloadする
        else:
            model_sub = VAE(l_i, l_o, act)
            chainer.serializers.load_npz(save_name, model_sub)
            model.append(model_sub)

    # 最後に全モデルを通して学習し直す
    if train_mode and len(model)>1:
        model = train_vae(model, train, epoch, batchsize)
        for i,(l_i, l_o) in enumerate(zip(layer[0:-1], layer[1:])):
            hidden_num = str(l_i) + '_' + str(l_o)
            save_name = os.path.join(folder_model, 'model_' + hidden_num + '.npz')
            chainer.serializers.save_npz(save_name, model[i])

    return model

