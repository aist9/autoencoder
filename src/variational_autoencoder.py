import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.computational_graph as c
from chainer import Variable, optimizers, initializer,cuda
from chainer.functions.loss.vae import gaussian_kl_divergence
import cupy
import os

# Xavier initializaer : used in class VAE
class Xavier(initializer.Initializer):
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


#Variational AutoEncoder model
class VAE(chainer.Chain):
    def __init__(self, layers, act_func=F.tanh):
        super(VAE, self).__init__()
        self.act_func = act_func
        self.makeLayers(layers)

    def makeLayers(self,hidden):
        # レイヤー作成. 作成順は任意だが,children()を使う関係上 nameが前伝搬の順に昇順でなければならない.
        # そのため、encoderは入力層に近いほど数字が若く、decoderはボトルネックに近いほど数字を若くする必要がある.
        for i in range(len(hidden)-2):
            # エンコード層を入力層からボトルネックの順で作成
            l = L.Linear(hidden[i],hidden[i+1], initialW=Xavier(hidden[i], hidden[i+1]))
            name  = 'enc{}'.format(i)
            self.add_link(name,l)
            # デコード層をボトルネックから出力層の順で作成
            j = len(hidden)-i-1
            l = L.Linear(hidden[j],hidden[j-1], initialW=Xavier(hidden[i], hidden[i+1]))
            name  = 'dec{}'.format(i)
            self.add_link(name,l)

        # μ,σを出力する層はここで定義
        self.add_link('enc_mu' ,L.Linear(hidden[-2],hidden[-1], initialW=Xavier(hidden[-2], hidden[-1])))
        self.add_link('enc_var',L.Linear(hidden[-2],hidden[-1], initialW=Xavier(hidden[-2], hidden[-1])))
        # 出力層はここで定義
        self.add_link('dec_out',L.Linear(hidden[1],hidden[0],   initialW=Xavier(hidden[1] , hidden[0] )))

        
    def __call__(self, x, sigmoid=True):
        mu,var = self.encode(x)
        e = F.gaussian(mu, var)
        d = self.decode(e,sigmoid)
        return e,d

    def encode(self, x):        
        e = x
        for layer in self.children():
            # エンコード層のみ処理
            if 'enc' in layer.name:
                if 'mu' in layer.name: # μ を出力するレイヤー
                    mu = layer(e)
                    continue
                if 'var' in layer.name: # σ を出力するレイヤー
                    var = layer(e)
                    break
                # 中間層の順伝搬と活性化
                e = self.act_func(layer(e))

        return mu,var

    def decode(self, z, sigmoid=True):
        d = z
        for layer in self.children():
            # デコード層のみ処理
            if 'dec' in layer.name:
                # 出力層は活性化しない
                if 'out' in layer.name: 
                    d = layer(d)
                    break
                d = self.act_func(layer(d))

        if sigmoid:
            d = F.sigmoid(d)
        return d


# 再構成と再構成誤差の計算
class Reconst():
    # 学習、モデルを渡しておく
    def __init__(self, model,sig=True):
        self.model = model
        self.sig = sig
    
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
        feat, reconst = self.model(Variable(data),self.sig)
        return feat.data, reconst.data

    def decode(self,data):
        reconst = self.model.decode(Variable(data),self.sig)
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

# 実際に学習する部分のコード
def train_vae(model,train,epoch,batch,C=1.0,k=1,use_sigmoid=False):

    model.to_gpu(0)
    opt = optimizers.Adam()
    opt.setup(model)

    # あらかじめgpuに投げつつVariable化する
    nlen = train.shape[0]
    data = Variable(cuda.to_gpu(train))

    # いつもの学習ループ
    for ep in range(epoch):
        perm = np.random.permutation(nlen)
        for p in range(0,nlen,batch):

            input_data = data[perm[p:p+batch]]
            model.cleargrads()
            # エンコード
            mu, ln_var = model.encode(input_data)

            rec_loss = 0
            for l in range(k): # 基本的にはk=1
                dec = F.gaussian(mu, ln_var)

                # デコード, bernoulli_nllがsigmoidを内包しているので,学習時は sigmoid = False
                reconst_data = model.decode(dec, sigmoid=False)

                rec_loss += F.bernoulli_nll(input_data,reconst_data) / (k * batch)
                # rec_loss += F.mean_squared_error(input_data,reconst_data) / (k)

            latent_loss = C * gaussian_kl_divergence(mu, ln_var) / batch
            loss = rec_loss + latent_loss

            # 逆伝搬
            loss.backward()
            opt.update()

        if (ep + 1) % 10 == 0:
            # print('\repoch ' + str(ep + 1) + ': ' + str(loss.data) + ' ', end = '')
            print('\repoch ' + str(ep + 1) + ': ' + str(loss.data) + ' ',latent_loss,rec_loss ,end = '\n')
            # print(model.a_enc0.W[0,0],model.a_enc1.W[0,0],model.a_enc2.W[0,0])
            # print(model.b_dec0.W[0,0],model.b_dec1.W[0,0],model.b_dec2.W[0,0])
    print()

    model.to_cpu()
    return model

def train_stacked(train, hidden, epoch, batchsize, folder, \
                  train_mode=True, \
                  act=F.tanh,C=1.0,k=1, use_sigmoid=False ):
 
    inputs = train.shape[1]
    if type(hidden) != type([]):
        hidden = [hidden]
    layer  = [inputs] + hidden
    
    # 隠れ層の数値を文字列にして保存・読み込みのフォルダを分ける
    hidden_str = []
    for i in range(len(hidden)):
        hidden_str.append(str(int(hidden[i])))
    hidden_num_str = '-'.join(hidden_str)
    print('layer' + str(layer))

    # 学習モデルの保存場所
    folder_model = os.path.join(folder, hidden_num_str)
    os.makedirs(folder_model, exist_ok=True)

    # モデルの保存名
    save_name = os.path.join(folder_model, 'model.npz')
    if train_mode:
        model = VAE(layer, act)
        model = train_vae(model, train, epoch, batchsize,C=C,k=k,use_sigmoid=use_sigmoid)
        chainer.serializers.save_npz(save_name, model)
    # 学習しない場合はloadする
    else:
        model = VAE(layer, act)
        chainer.serializers.load_npz(save_name, model)

    return model

