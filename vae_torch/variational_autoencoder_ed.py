# encoder と decoderが独立しているバージョン
# 無印版との大きな違いはエンコーダとデコーダが異なるoptimizerを持つことだが, 現状それを特に活かした実装はしていない
# なんとなく学習が不安定なイメージがある
# およそvariational_autoencoderを継承

import numpy as np
import os
import matplotlib.pyplot as plt
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from torch.utils.data import DataLoader
from ignite.engine import Engine, Events
from ignite.utils import convert_tensor
from ignite.contrib.handlers.tensorboard_logger import *

from variational_autoencoder import VAE

log2pi = float(np.log(2*np.pi))


#Variational AutoEncoder model
class Encoder(nn.Module):
    def __init__(self, layers, act_func=torch.tanh, use_BN=False, init_method=nn.init.xavier_uniform_, device='cuda'):
        super(Encoder, self).__init__()
        self.use_BN = use_BN
        self.act_func = act_func
        self.device = device

        self._makeLayers(layers,init_method)


    def _makeLayers(self,hidden, init_method):
        encode_layer = []

        for i in range(len(hidden)-2):
            e = nn.Linear(hidden[i],hidden[i+1])
            init_method(e.weight)
            encode_layer.append( e )
            if self.use_BN:
                e = nn.BatchNorm1d(hidden[i+1])
                encode_layer.append( e )

        # pytorchでは↓のように書けば重み更新されるらしい
        self.encode_layer = nn.ModuleList(encode_layer)
        # μ,σを出力する層はここで定義
        self.enc_mu  = nn.Linear(hidden[-2],hidden[-1])
        init_method(self.enc_mu.weight)
        self.enc_var = nn.Linear(hidden[-2],hidden[-1])
        init_method(self.enc_var.weight)


    def __call__(self, x):
        mu,var = self.encode(x)
        e = self.sample_z(mu, var)
        return mu, var, e


    def encode(self, x):
        e = x
        for i in range(len(self.encode_layer)):
            # BatchNormレイヤーの直前では活性化しない (BNでないレイヤーは0から数えて偶数番目)
            e = self.encode_layer[i](e) if self.use_BN and not (i&1) else self.act_func(self.encode_layer[i](e))
            
        mu  = self.enc_mu(e)
        var = self.enc_var(e)
        # var = F.softplus(var) # varにsoftplusをかける実装を見かけたので一応
        return mu,var

    # 平均と分散からガウス分布を生成. 実装パクっただけなのでよくわからん
    def sample_z(self, mu, var):
        epsilon = torch.randn(mu.shape).to(self.device)
        return mu + torch.sqrt(torch.exp(var)) * epsilon

    def latent_loss(self, mu, var):
        return 0.5 *  torch.mean(torch.sum( -var -1  +mu*mu +torch.exp(var), dim=1))


class Decoder(nn.Module):
    def __init__(self, layers, act_func=torch.tanh, out_func=torch.sigmoid, use_BN=False, init_method=nn.init.xavier_uniform_, is_gauss_dist=False, device='cuda'):
        super(Decoder, self).__init__()
        self.use_BN = use_BN
        self.act_func = act_func
        self.out_func = out_func
        self.is_gauss_dist = is_gauss_dist
        self.device = device

        self._makeLayers(layers,init_method)


    def _makeLayers(self,hidden, init_method):
        decode_layer = []

        for i in range(len(hidden)-2):
            j = len(hidden)-i-1
            d = nn.Linear(hidden[j],hidden[j-1])
            init_method(d.weight)
            decode_layer.append( d )
            if self.use_BN and j>1:
                d = nn.BatchNorm1d(hidden[j-1])
                decode_layer.append( d )

        self.decode_layer = nn.ModuleList(decode_layer)
        # 出力層はここで定義. dec_muとしているが, bernoulliバージョンではmuを出力するわけではない. dec_out1とかのほうがいいかも
        self.dec_mu  = nn.Linear(hidden[1],hidden[0])
        init_method(self.dec_mu.weight)
        if self.is_gauss_dist:
            self.dec_var = nn.Linear(hidden[1],hidden[0])
            init_method(self.dec_var.weight)

        
    def __call__(self, z):
        d_out = self.decode(z)
        return d_out

    def decode(self, z):
        d = z
        for i in range(len(self.decode_layer)):
            d = self.decode_layer[i](d) if self.use_BN and not (i&1) else self.act_func(self.decode_layer[i](d))

        # gauss_distバージョンなら2出力, bernoulliバージョンなら1出力
        # d_out = ( self.out_func(self.dec_mu(d)), self.out_func(self.dec_var(d)) ) if self.is_gauss_dist else self.out_func(self.dec_mu(d))
        d_out = (self.dec_mu(d), self.out_func(self.dec_var(d)) ) if self.is_gauss_dist else self.out_func(self.dec_mu(d))
        return d_out

    def reconst_loss(self, x, dec_out):
        if self.is_gauss_dist:
            dec_mu, dec_var = dec_out
            m_vae = 0.5* (x - dec_mu)**2 * torch.exp(-dec_var)
            a_vae = 0.5* (log2pi+dec_var)
            reconst = torch.mean( torch.sum(m_vae + a_vae, dim=1) )
        else:
            reconst = -torch.mean(torch.sum(x * torch.log(dec_out) + (1 - x) * torch.log(1 - dec_out), dim=1))

        return reconst


class VAE_ED(VAE):
    # コンストラクタは継承したものをそのまま使用
    # def __init__(self, input_shape, hidden, act_func=torch.tanh, use_BN=False, init_method='xavier', folder='./model', is_gauss_dist=False, device='cuda'):

    # 学習, 継承したものをそのまま使用
    # def train(self,train,epoch,batch,C=1.0, k=1, valid=None, is_plot_weight=False):

    # trainer, 継承したものをそのまま使用
    # def trainer(self, C=1.0, k=1, device=None):
    

    # 多分使う機会はほとんどない
    def __call__(self, x):
        # self.encode ではなく, encoderのコール関数を呼び出していることに注意
        mu, var, e = self.encoder(x)
        d_out = self.decoder(e) # なんとなくエンコーダに合わせる
        return e, d_out


    def encode(self, x):
        return self.encoder.encode(x)

    def decode(self, z):
        return self.decoder.decode(z)

    def latent_loss(self, mu, var):
        return self.encoder.latent_loss(mu, var)

    def reconst_loss(self, x, d_out):
        return self.decoder.reconst_loss(x, d_out)

    def set_model(self, hidden, act_func, out_func, use_BN, init_method, is_gauss_dist, device):
        self.encoder = Encoder(hidden, act_func, use_BN, init_method, device=device)
        self.decoder = Decoder(hidden, act_func, out_func, use_BN, init_method, is_gauss_dist=is_gauss_dist, device=device)

    def set_optimizer(self, learning_rate=0.001, beta1=0.9, beta2=0.999, weight_decay=0, gradient_clipping=None):
        betas=(beta1, beta2)
        self.opt_enc = optim.Adam(self.encoder.parameters(), lr=learning_rate, betas=betas, eps=1e-08, weight_decay=weight_decay, amsgrad=False)
        self.opt_dec = optim.Adam(self.decoder.parameters(), lr=learning_rate, betas=betas, eps=1e-08, weight_decay=weight_decay, amsgrad=False)
        self.gradient_clipping = gradient_clipping

    def model_to(self, device):
        self.encoder.to(device)
        self.decoder.to(device)

    def sample_z(self, mu, var):
        return self.encoder.sample_z(mu, var)

    def zero_grad(self):
        self.opt_enc.zero_grad()
        self.opt_dec.zero_grad()

    def step(self):
        self.opt_enc.step()
        self.opt_dec.step()

    def grad_clip(self):
        if self.gradient_clipping is not None:
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.gradient_clipping)
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.gradient_clipping)


    # 各レイヤーの重みをプロット. 重み更新が機能してるか確認.
    def plot_weight(self, epoch):
        enc_dic = self.encoder.state_dict()
        dec_dic = self.decoder.state_dict()

        fig = plt.figure(figsize=(16,8))
        plot_num = 0
        for dic in [enc_dic, dec_dic]:
            for k in dic.keys():
                if 'weight' in k:
                    plot_num += 1
                    plot_data = self.tensor_to_np(dic[k]).reshape(-1)
                    plt.subplot(2,self.weight_num,plot_num)
                    plt.plot(plot_data, label=k)
                    plt.legend()
        plt.tight_layout()
        plt.savefig( self.save_dir + '/weight_plot/epoch_{}.png'.format(epoch) )
        plt.close()


    # modelの保存. trainメソッドの最後に呼び出される.
    def save_model(self, path=None):
        path = self.save_dir if path is None else path
        torch.save(self.encoder.state_dict(), path+'/encoder.pth')
        torch.save(self.decoder.state_dict(), path+'/decoder.pth')

    # modelのロード. 学習済みモデルを使用したい場合に呼び出す.
    def load_model(self, path=None):
        path = self.save_dir if path is None else path
        enc_param = torch.load( path + '/encoder.pth')
        dec_param = torch.load( path + '/decoder.pth')
        self.encoder.load_state_dict(enc_param)
        self.decoder.load_state_dict(dec_param)
        self.model_to(self.device)


    ###### 評価関係 #########

    # evalモードに切り替え
    def model_to_eval(self):
        self.encoder.eval()
        self.decoder.eval()

    # 以下, 継承
    # 入力データの 潜在特徴, 再構成データ, 誤差 を返す.
    # def reconst(self, data, unregular=False):

    # 再構成誤差を二乗平均で計算. 主に学習時の確認用.
    # def MSE(self, data):

    # 潜在特徴を入力したときのデコーダの出力を返す
    # def featuremap_to_image(self, feat):

    # def np_to_tensor(self, data):
    #     return torch.Tensor(data).to(self.device)

    # def tensor_to_np(self, data):
    #     return data.detach().to('cpu').numpy()



