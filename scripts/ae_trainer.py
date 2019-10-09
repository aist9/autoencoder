
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

# オートエンコーダクラス
class Autoencoder(Chain):
    def __init__(self, inputs, hidden, fe='sigmoid', fd='sigmoid'):
        super(Autoencoder, self).__init__()
        with self.init_scope():
            self.le = L.Linear(inputs, hidden)
            self.ld = L.Linear(hidden, inputs)

        # 活性化関数の指定
        self.fe = fe
        self.fd = fd

    # Forward
    # hidden_out=Trueにすると隠れ層出力もreturn
    def __call__(self, x, hidden_out=False):
        # h = F.dropout(self.encoder(x))
        # y = F.dropout(self.decoder(h))
        h = (self.encoder(x))
        y = (self.decoder(h))
        if hidden_out == False:
            return y
        else:
            return y, h

    # Encoder
    def encoder(self, x):
        h = self.le(x)
        if self.fe is not None:
            h = self.activation_function(h, self.fe)
        return h
        
    # Decode
    def decoder(self, h):
        y = self.ld(h)
        if self.fd is not None:
            y = self.activation_function(y, self.fd) 
        return y
    
    # 活性化関数
    def activation_function(self, data, func): 
        if func == 'tanh':
            data = F.tanh(data)
        elif func == 'sigmoid':
            data = F.sigmoid(data)
        elif func == 'relu':
            data = F.relu(data)
        elif func == 'leaky':
            data = F.leaky_relu(data)
        return data

# trainer使うためのラッパー
class AutoencoderTrainer(Chain):
    def __init__(self, ae, ae_method=None, rho=0.05, s=0.001):
        super(AutoencoderTrainer, self).__init__(ae=ae)

        # AEの種類を指定
        self.ae_method = ae_method

        # Sparse AEの平均活性化度と正則化の強さ
        self.rho = rho
        self.s = s

    # trainerで呼ばれるcall関数
    def __call__(self, x, t):
        # Forward
        if self.ae_method == 'sparse':
            # sparsei AEの場合は隠れ層出力を受け取りKLDを計算する
            y, h = self.ae(x, hidden_out=True)
            kld = self.reg_sparse(h)
            loss = F.mean_squared_error(y, t) + self.s * kld
        else:
            y = self.ae(x)
            loss = F.mean_squared_error(y, t)
            # ma = np.argmax( abs(chainer.cuda.to_cpu((y-t).data)), axis=1  )
            # loss += F.mean_absolute_error(y[:,ma],t[:,ma])
            # loss = F.mean_absolute_error(y,t)
        
        # Chainerのreport機能
        reporter.report({'loss': loss}, self)

        return loss

    # Sparse正則化項の計算
    def reg_sparse(self, h):
        rho_hat = F.sum(h, axis=0) / h.shape[0]
        kld = F.sum(self.rho * F.log(self.rho / rho_hat) + \
                    (1 - self.rho) * F.log((1 - self.rho) / (1 - rho_hat)))
        return kld


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
        for i in range(self.L):
            feat = self.model[i].encoder(feat)
        
        reconst = feat
        for i in range(self.L):
            reconst = self.model[self.L - i - 1].decoder(reconst)

        return feat.data, reconst.data

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
    
    # しきい値を超えた場合はTrue
    def judgment(self, err, th):
        result = [False if err[i] <= th else True for i in range(len(err))]
        return result

# オーエンコーダの学習(trainer)
def train_ae(data, hidden, max_epoch, batchsize, \
             fe='sigmoid', fd='sigmoid', gpu_device=-1, \
             ae_method=None, rho=0.05, s=0.001):
    
    # 入力サイズ
    inputs = data.shape[1]
    # データの数
    len_train = data.shape[0]

    # モデルの定義
    ae = Autoencoder(inputs, hidden, fe=fe, fd=fd)
    model = AutoencoderTrainer(ae, ae_method=ae_method, rho=rho, s=s)
    opt = optimizers.Adam()
    opt.setup(model)

    # データの形式を変換する
    train = datasets.TupleDataset(data, data)
    train_iter = iterators.SerialIterator(train, batchsize)

    # 学習ループ
    updater = training.StandardUpdater(train_iter, opt, device=gpu_device)
    #updater = training.ParallelUpdater(train_iter, opt, devices={'main': 0, 'second': 1})
    trainer = training.Trainer(updater, (max_epoch, 'epoch'), out="result")
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport( ['epoch', 'main/loss']))
    trainer.run()

    # GPUを使っていた場合CPUに戻す
    if -1 < gpu_device:
        ae.to_cpu()

    return ae

# Stacked AutoEncoderの学習
#     stackedでなくても学習してくれる
#     folderで指定した場所に各層の学習モデルを保存してくれる
#     train_modeがFalseのときはfolderからモデルを読み込み
def train_stacked(train, hidden, epoch, batchsize, folder, \
                  train_mode=True, \
                  fe='sigmoid', fd='sigmoid', \
                  ae_method=None, rho=[0.05], s=[0.001]):
 
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
    feat = train
    for i,(l_i, l_o) in enumerate(zip(layer[0:-1], layer[1:])):
        # 保存に使う文字列
        hidden_num = str(l_i) + '_' + str(l_o)
        # モデルの保存名
        save_name = os.path.join(folder_model, 'model_' + hidden_num + '.npz')

        act_enc = fe
        act_dec = fd
        if type(fe) == type([]):
            act_enc = fe[i]
        if type(fd) == type([]):
            act_dec = fd[i]

        # 学習を行いsaveする
        if train_mode == True or not os.path.isfile(save_name):
            # 学習を行い、リストにappendしていく
            print("Layer ", i + 1)
            rho_ = rho[i]
            s_ = s[i]
            model_sub = train_ae(feat, l_o, epoch, batchsize, fe=act_enc, fd=act_dec,\
                                 gpu_device=1, ae_method=ae_method, rho=rho_, s=s_)
            model.append(model_sub)
            feat = model_sub.encoder(Variable(feat)).data

            # モデルの保存
            chainer.serializers.save_npz(save_name, model_sub)

        # 学習しない場合はloadする
        else:
            model_sub = Autoencoder(l_i, l_o, fe=act_enc, fd=act_dec)
            chainer.serializers.load_npz(save_name, model_sub)
            model.append(model_sub)

    return model

