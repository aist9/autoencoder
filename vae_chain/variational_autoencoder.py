import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable, optimizers, initializer,cuda
import os
import matplotlib.pyplot as plt
import cupy as cp

from net import Net, Xavier

#Variational AutoEncoder model
class Encoder_Decoder(chainer.Chain):
    def __init__(self, layers, act_func=F.tanh, out_func=F.sigmoid, use_BN=False, init_method='xavier', is_gauss_dist=False):
        super(Encoder_Decoder, self).__init__()
        self.use_BN = use_BN
        self.act_func = act_func
        self.out_func = out_func
        self. is_gauss_dist=is_gauss_dist

        self.makeLayers(layers,init_method)


    def makeLayers(self,hidden, init_method):
        for i in range(len(hidden)-2):
            init_func = Xavier(hidden[i], hidden[i+1]) if init_method=='xavier' else init_method
            l = L.Linear(hidden[i],hidden[i+1], init_func)
            name  = 'enc{}'.format(i)
            self.add_link(name,l)
            if self.use_BN:
                l = L.BatchNormalization(hidden[i+1])
                name  = 'enc{}_BN'.format(i)
                self.add_link(name,l)

            j = len(hidden)-i-1
            l = L.Linear(hidden[j],hidden[j-1], initialW=init_func)
            name  = 'dec{}'.format(i)
            self.add_link(name,l)
            if self.use_BN and j>1:
                l = L.BatchNormalization(hidden[j-1])
                name  = 'dec{}_BN'.format(i)
                self.add_link(name,l)

        if init_method == 'xavier':
            # μ,σを出力する層はここで定義
            self.add_link('enc_mu' ,L.Linear(hidden[-2],hidden[-1], initialW=Xavier(hidden[-2], hidden[-1])))
            self.add_link('enc_var',L.Linear(hidden[-2],hidden[-1], initialW=Xavier(hidden[-2], hidden[-1])))
            # 出力層はここで定義
            self.add_link('dec_out1' ,L.Linear(hidden[1],hidden[0],  initialW=Xavier(hidden[1] , hidden[0] )))
            if self.is_gauss_dist:
                self.add_link('dec_out2' ,L.Linear(hidden[1],hidden[0],  initialW=Xavier(hidden[1] , hidden[0] )))
        else:
            self.add_link('enc_mu' ,L.Linear(hidden[-2],hidden[-1], initialW=init_method))
            self.add_link('enc_var',L.Linear(hidden[-2],hidden[-1], initialW=init_method))
            self.add_link('dec_out1' ,L.Linear(hidden[1],hidden[0],  initialW=init_method))
            if self.is_gauss_dist:
                self.add_link('dec_out2' ,L.Linear(hidden[1],hidden[0],  initialW=init_method))

        
    def __call__(self, x):
        mu,var = self.encode(x)
        e = F.gaussian(mu, var)
        d_out = self.decode(e)
        return e,d_out

    def encode(self, x):        
        e = x
        for layer in self.children():
            if 'enc' in layer.name:
                if 'mu' in layer.name: # μ を出力するレイヤー
                    mu = layer(e)
                    continue
                if 'var' in layer.name: # σ を出力するレイヤー
                    var = layer(e)
                    break                
                e = layer(e)
                e = self.act_func(e) if ('BN' in layer.name or not self.use_BN) else e

        return mu,var

    def decode(self, z):
        d = z
        for layer in self.children():
            if 'dec' in layer.name:
                if 'out' in layer.name: # 出力レイヤー
                    break
                d = layer(d)
                d = self.act_func(d) if ('BN' in layer.name or not self.use_BN) else d

        # out = ( self.out_func(self.dec_out1(d)), self.out_func(self.dec_out2(d)) ) if self.is_gauss_dist else self.dec_out1(d)
        out = ( self.dec_out1(d), self.out_func(self.dec_out2(d)) ) if self.is_gauss_dist else self.dec_out1(d)
        return out


class VAE(Net):
    # def __init__(self, input_shape, hidden, act_func=F.tanh, out_func=F.sigmoid ,use_BN=False, init_method='xavier', folder='./model', is_gauss_dist=False):
        
    # 多分使う機会はほとんどない
    # def __call__(self, x):    

    # 学習
    # def train(self,train,epoch,batch,C=1.0,k=1, gpu_num=0,valid=None, is_plot_weight=False):

    def encode(self, encoder_input):
        return self.model.encode(encoder_input)

    def decode(self, decoder_input):
        return self.model.decode(decoder_input)

    def cleargrads(self):
        self.model.cleargrads()

    def update(self):
        self.opt.update()

    def set_model(self, hidden, act_func, out_func, use_BN, init_method, is_gauss_dist):
        self.model = Encoder_Decoder(hidden, act_func=act_func, out_func=out_func, use_BN=use_BN, init_method=init_method, is_gauss_dist=is_gauss_dist)

    def set_optimizer(self, learning_rate=0.001, gradient_momentum=0.9, weight_decay=None, gradient_clipping=None):
        self.opt = optimizers.Adam(alpha=learning_rate, beta1=gradient_momentum)
        # self.opt = optimizers.Adam()
        self.opt.setup(self.model)

        if gradient_clipping is not None:
            self.opt.add_hook(chainer.optimizer.GradientClipping(gradient_clipping))
        if weight_decay is not None:
            self.opt.add_hook(chainer.optimizer.WeightDecay(0.001))

    def model_to(self, gpu_num):
        if gpu_num == -1:
            self.model.to_cpu()
        else:
            self.model.to_gpu(gpu_num)


    # 各レイヤーの重みをプロット. 重み更新が機能してるか確認.
    def plot_weight(self, epoch):
        
        j = 0
        fig = plt.figure(figsize=(16,8))
        for layer in self.model.children():
            if 'BN' in layer.name:
                continue
            j+=1
            plt.subplot(2,self.weight_num,j)
            plt.plot(cp.asnumpy(layer.W.data).reshape(-1), label=layer.name)
            plt.legend()

        plt.savefig( self.save_dir + '/weight_plot/epoch_{}.png'.format(epoch+1) )
        plt.close()


    # modelの保存. trainメソッドの最後に呼び出される. ついでにエラーカーブも保存.
    def save_model(self, path=None, train_loss=None):
        path = self.save_dir if path is None else path
        chainer.serializers.save_npz(path+'/model.npz', self.model)
        if train_loss is not None:
            fig = plt.figure(figsize=(12,8))
            plt.plot(train_loss)
            plt.savefig(path+'/error_curve.png')
            plt.close()

    # modelのロード. 学習済みモデルを使用したい場合に呼び出す.
    def load_model(self, path=None):
        path = self.save_dir if path is None else path
        chainer.serializers.load_npz(path+'/model.npz', self.model)

    # 入力データの 潜在特徴, 再構成データ, 誤差 を返す.
    # def reconst(self, data, unregular=False):

    # 再構成誤差を二乗平均で計算. 主に学習時の確認用.
    # def MSE(self, data):

    # 潜在特徴を入力したときのデコーダの出力を返す
    # def featuremap_to_image(self, feat):


