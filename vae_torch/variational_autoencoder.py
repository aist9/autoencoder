# pytorchを使ったVAEの実装
# ae_torchとは使用方法が異なる. その内統一したい
# class VAEについて, 余分に思えるメソッドがあるがこれは継承することを考慮して書いている

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

log2pi = float(np.log(2*np.pi))


#Variational AutoEncoder model
class Encoder_Decoder(nn.Module):
    def __init__(self, layers, act_func=torch.tanh, out_func=torch.sigmoid, use_BN=False, init_method=nn.init.xavier_uniform_, is_gauss_dist=False ,device='cuda'):
        super(Encoder_Decoder, self).__init__()
        self.use_BN = use_BN
        self.act_func = act_func
        self.out_func = out_func
        self.is_gauss_dist = is_gauss_dist
        self.device = device

        self._makeLayers(layers,init_method)

    def _makeLayers(self,hidden, init_method):
        encode_layer = []
        decode_layer = []

        for i in range(len(hidden)-2):
            # set encoder layer
            e = nn.Linear(hidden[i],hidden[i+1])
            init_method(e.weight)
            encode_layer.append( e )
            if self.use_BN:
                e = nn.BatchNorm1d(hidden[i+1])
                encode_layer.append( e )

            # set decoder layer
            j = len(hidden)-i-1
            d = nn.Linear(hidden[j],hidden[j-1])
            init_method(d.weight)
            decode_layer.append( d )
            if self.use_BN and j>1:
                d = nn.BatchNorm1d(hidden[j-1])
                decode_layer.append( d )

        # pytorchでは↓のように書けば重み更新されるらしい
        self.encode_layer = nn.ModuleList(encode_layer)
        self.decode_layer = nn.ModuleList(decode_layer)
        # enc と dec で分けるか諸説ある
        # self.layer = nn.ModuleList(encode_layer+decode_layer)

        # μ,σを出力する層はここで定義
        self.enc_mu  = nn.Linear(hidden[-2],hidden[-1])
        init_method(self.enc_mu.weight)
        self.enc_var = nn.Linear(hidden[-2],hidden[-1])
        init_method(self.enc_var.weight)

        # 出力層はここで定義. dec_muとしているが, bernoulliバージョンではmuを出力するわけではない. dec_out1とかのほうがいいかも
        self.dec_mu  = nn.Linear(hidden[1],hidden[0])
        init_method(self.dec_mu.weight)
        if self.is_gauss_dist:
            self.dec_var = nn.Linear(hidden[1],hidden[0])
            init_method(self.dec_var.weight)

        
    def __call__(self, x):
        mu,var = self.encode(x)
        e = self.sample_z(mu, var)
        d_out = self.decode(e)
        return e,d_out

    def encode(self, x):
        e = x
        for i in range(len(self.encode_layer)):
            # BatchNormレイヤーの直前では活性化しない (BNでないレイヤーは0から数えて偶数番目)
            e = self.encode_layer[i](e) if self.use_BN and not (i&1) else self.act_func(self.encode_layer[i](e))
            
        mu  = self.enc_mu(e)
        var = self.enc_var(e)
        # var = F.softplus(var) # varにsoftplusをかける実装を見かけたので一応
        return mu,var

    def decode(self, z):
        d = z
        for i in range(len(self.decode_layer)):
            d = self.decode_layer[i](d) if self.use_BN and not (i&1) else self.act_func(self.decode_layer[i](d))

        # gaussバージョンなら2出力, bernoulliバージョンなら1出力
        # d_out = ( self.out_func(self.dec_mu(d)) , self.out_func(self.dec_var(d)) ) if self.is_gauss_dist else self.out_func(self.dec_mu(d))
        d_out = ( self.dec_mu(d), self.out_func(self.dec_var(d)) ) if self.is_gauss_dist else self.out_func(self.dec_mu(d))
        return d_out

        # gaussバージョンでの出力の活性化関数の有無について(MNISTで実験, sigmoidでのみ試行)
        # bernoulli分布と異なり, gaussian分布では平均・分散が0~1である必要はないはずだが, MNISTで試行するとsigmoidをかけたほうが学習がうまく行く.
        # dec_mu, dec_var ともに無し... ロスが負の方向に増大していく. 潜在特徴や再構成の結果は上手くいかない.
        # dec_mu, dec_var ともに有り... ロスは正の方向に減少していく. 潜在特徴でクラス判別可. 再構成可.
        # dec_mu  のみ有り          ... ロスは正の方向に増大していく. nanが出るため学習不可.
        # dec_var のみ有り          ... ロスは正の方向に減少していく. 潜在特徴でクラス判別可. 再構成可.
        # 結論 : 入力を標準化(平均0分散1)した結果, dec_varにsigmoidをかけた場合のみ正常に機能したためこれを標準実装とする(eluやsoftplusでも良さそう？).
        # dec_ln_varとして学習できていない？ ln_varとして学習できているなら負の値を許容できるはずだが, sigmoidで負の値を弾いているから学習が上手くいっている気がする.

    # 平均と分散からガウス分布を生成. 実装パクっただけなのでよくわからん. encoderはvarの代わりにln(var)を出力させる 
    def sample_z(self, mu, ln_var):
      epsilon = torch.randn(mu.shape).to(self.device)
      return mu + torch.exp(ln_var*0.5) * epsilon
      # return mu + torch.sqrt( torch.exp(ln_var) ) * epsilon


    def latent_loss(self, enc_mu, enc_var):
        return -0.5 *  torch.mean(torch.sum( enc_var +1  -enc_mu*enc_mu -torch.exp(enc_var), dim=1))

    def reconst_loss(self, x, dec_out):
        if self.is_gauss_dist:
            dec_mu, dec_var = dec_out
            m_vae = 0.5* (x - dec_mu)**2 * torch.exp(-dec_var)
            a_vae = 0.5* (log2pi+dec_var)
            reconst = torch.mean( torch.sum(m_vae + a_vae, dim=1) )
        else:
            reconst = -torch.mean(torch.sum(x * torch.log(dec_out) + (1 - x) * torch.log(1 - dec_out), dim=1))

        return reconst



class VAE():
    def __init__(self, input_shape, hidden, act_func=torch.tanh, out_func=torch.sigmoid, use_BN=False, init_method='xavier', folder='./model', is_gauss_dist=False, device='cuda'):

        activations = {
                "sigmoid"   : torch.sigmoid, \
                "tanh"      : torch.tanh,    \
                "softplus"  : F.softplus,    \
                "relu"      : torch.relu,    \
                "leaky"     : F.leaky_relu,  \
                "elu"       : F.elu,         \
                "identity"  : lambda x:x     \
        }

        # gpu setting
        self.device = device
        
        # 活性化関数 文字列で指定されたとき関数に変換
        if isinstance(act_func, str):
            if act_func in activations.keys():
                act_func = activations[act_func]
            else:
                print('arg act_func is ', act_func, '. This value is not exist. This model uses identity function as activation function.')
                act_func = lambda x:x

        if isinstance(out_func, str):
            if out_func in activations.keys():
                out_func = activations[out_func]
            else:
                print('arg out_func is ', out_func, '. This value is not exist. This model uses identity function as activation function of output.')
                out_func = lambda x:x
        if out_func != torch.sigmoid:
            print('※ out_func should be sigmoid.')

                
        # 重みの初期化手法 文字列で指定されたとき関数に変換
        if isinstance(init_method, str):
            inits = {
                    "xavier"    : nn.init.xavier_uniform_, \
                    "henormal"  : nn.init.kaiming_uniform_ \
            }
            if init_method in inits.keys():
                init_method = inits[init_method]
            else:
                init_method = nn.init.xavier_uniform_
                print('init_method is xavier initializer')


        if not isinstance(hidden, list):
            hidden = [hidden]
        hidden = [input_shape] + hidden
        print('layer' + str(hidden))

        # model保存用のパス
        self.save_dir  = os.path.join(folder, '{}'.format(hidden))
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.save_dir+'/weight_plot', exist_ok=True)

        self.is_gauss_dist = is_gauss_dist
        self.set_model(hidden, act_func, out_func, use_BN, init_method, is_gauss_dist=is_gauss_dist, device=self.device)
        self.set_optimizer()

        # encoderの重みの数. weight_plotで使用.
        self.weight_num = len(hidden) + 1

    # 多分使う機会はほとんどない
    def __call__(self, x):
        return self.model(x)


    # train
    def train(self,train,epoch,batch,C=1.0, k=1, valid=None, is_plot_weight=False):

        if valid is None:
            print('epoch\tloss\t\tlatent\t\treconst\t\tMSE')
        else:
            print('epoch\tloss\t\tlatent\t\treconst\t\tMSE\t\tvalid_MSE')


        # conversion data
        train_data = torch.Tensor(train)
        dataset = torch.utils.data.TensorDataset(train_data, train_data)
        train_loader = DataLoader(dataset, batch_size=batch, shuffle=True)

        # trainer
        trainer = self.trainer(C=C,k=k, device=self.device)

        # log variables init.
        log = []
        loss_iter = []
        lat_loss_iter = []
        rec_loss_iter = []

        # executed function per iter
        @trainer.on(Events.ITERATION_COMPLETED)
        def add_loss(engine):
            loss_iter.append(engine.state.output[0])
            lat_loss_iter.append(engine.state.output[1])
            rec_loss_iter.append(engine.state.output[2])
        
        # executed function per epoch
        @trainer.on(Events.EPOCH_COMPLETED)
        def log_report(engine):
            epoch = engine.state.epoch
            loss     = np.mean(loss_iter)
            lat_loss = np.mean(lat_loss_iter)
            rec_loss = np.mean(rec_loss_iter)
            log.append({'epoch':epoch,'loss':loss, 'latent':lat_loss, 'reconst':rec_loss})
            if epoch % 10 == 0 or epoch==1:
                perm = np.random.permutation(len(train))[:batch]
                mse = self.MSE(train[perm]).mean()
                if valid is None:
                    print(f'{epoch}\t{loss:.6f}\t{lat_loss:.6f}\t{rec_loss:.6f}\t{mse:.6f}')
                else:
                    val_mse = self.MSE(valid).mean()
                    print(f'{epoch}\t{loss:.6f}\t{lat_loss:.6f}\t{rec_loss:.6f}\t{mse:.6f}\t{val_mse:.6f}')

                if is_plot_weight: # output layer weight.
                    self.plot_weight(epoch)

            loss_iter.clear()
            rec_loss_iter.clear()
            lat_loss_iter.clear()

        # start training
        trainer.run(train_loader, max_epochs=epoch)

        # save model weight
        self.save_model()
        
        # log output
        file_path = os.path.join(self.save_dir, 'log')
        file_ = open(file_path, 'w')
        json.dump(log, file_, indent=4)
    

    def trainer(self, C=1.0, k=1, device=None):

        self.model_to(device)

        def prepare_batch(batch, device=None):
            x, y = batch
            return (convert_tensor(x, device=device),
                    convert_tensor(y, device=device))

        def _update(engine, batch):
            self.zero_grad()
            x,y = prepare_batch(batch, device=device)
            e_mu, e_lnvar = self.encode(x)
            latent_loss = self.latent_loss(e_mu, e_lnvar)
            reconst_loss = 0
            for l in range(k):
                z = self.sample_z(e_mu, e_lnvar)
                d_out = self.decode(z)
                reconst_loss += self.reconst_loss(y, d_out) / float(k)
            loss = latent_loss + reconst_loss
            loss.backward()
            self.grad_clip()
            self.step()
            return loss.item(), latent_loss.item(), reconst_loss.item()
 
        return Engine(_update)


    # 継承時の記述省略のためのオーバーライド用
    def encode(self, x):
        return self.model.encode(x)

    def decode(self, z):
        return self.model.decode(z)

    def latent_loss(self, mu, var):
        return self.model.latent_loss(mu, var)

    def reconst_loss(self, x, d_out):
        return self.model.reconst_loss(x, d_out)

    def set_model(self, hidden, act_func, out_func, use_BN, init_method, is_gauss_dist, device):
        self.model = Encoder_Decoder(hidden, act_func, out_func, use_BN, init_method, is_gauss_dist=is_gauss_dist, device=device)

    def set_optimizer(self, learning_rate=0.001, beta1=0.9, beta2=0.999, weight_decay=0, gradient_clipping=None):
        betas=(beta1, beta2)
        self.opt = optim.Adam(self.model.parameters(), lr=learning_rate, betas=betas, eps=1e-08, weight_decay=weight_decay, amsgrad=False)
        self.gradient_clipping = gradient_clipping


    # モデルをcpuにしたりcudaにしたりするやつ
    def model_to(self, device):
        self.model.to(device)

    # trainメソッドで model.sample_zって書きたくなかった
    def sample_z(self, mu, var):
        return self.model.sample_z(mu, var)

    def zero_grad(self):
        self.opt.zero_grad()

    def step(self):
        self.opt.step()

    def grad_clip(self):
        if self.gradient_clipping is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)

    # 各レイヤーの重みをプロット. 重み更新が機能してるか確認.
    def plot_weight(self, epoch):
        dic = self.model.state_dict()

        fig = plt.figure(figsize=(16,8))
        plot_num = 0
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
        torch.save(self.model.state_dict(), path+'/model.pth')

    # modelのロード. 学習済みモデルを使用したい場合に呼び出す.
    def load_model(self, path=None):
        path = self.save_dir if path is None else path
        param = torch.load( path + '/model.pth')
        self.model.load_state_dict(param)
        self.model.to(self.device)


    ###### 評価関係 #########

    # evalモードに切り替え
    def model_to_eval(self):
        self.model.eval()

    # 入力データの 潜在特徴, 再構成データ, 誤差 を返す.
    def reconst(self, data, unregular=False):

        if data.ndim == 1:
            data = data.reshape(1,-1)
        if not isinstance(data, torch.Tensor):
            data = self.np_to_tensor(data)

        e_mu,e_var = self.encode(data)
        feat  = self.sample_z(e_mu, e_var)
        d_out = self.decode(feat)

        rec = d_out[0] if self.is_gauss_dist else d_out
        mse = torch.mean( (rec-data)**2, dim=1 )

        # lat_loss = self.latent_loss(  e_mu, e_var)
        # rec_loss = self.reconst_loss( data, d_out )
        # vae_err = (lat_loss+rec_loss)

        feat = self.tensor_to_np(feat)
        rec  = self.tensor_to_np(rec)
        mse  = self.tensor_to_np(mse)

        return feat, rec, mse

    # 再構成誤差を二乗平均で計算. 主に学習時の確認用.
    def MSE(self, data):
        if data.ndim == 1:
            data = data.reshape(1,-1)
        if not isinstance(data, torch.Tensor):
            data = self.np_to_tensor(data)

        e, d_out = self(data)
        rec = d_out[0] if self.is_gauss_dist else d_out
        mse = torch.mean( (rec-data)**2, dim=1 )
        return self.tensor_to_np(mse)

    # 潜在特徴を入力したときのデコーダの出力を返す
    def featuremap_to_image(self, feat):
        if feat.ndim == 1:
            feat = feat.reshape(1,-1)
        if not isinstance(feat, torch.Tensor):
            feat = self.np_to_tensor(feat)

        d_out = self.decode(feat)
        d_out = d_out[0] if self.is_gauss_dist else d_out
        return self.tensor_to_np(d_out)

    # ndarray -> torch.Tensor 
    def np_to_tensor(self, data):
        return torch.Tensor(data).to(self.device)

    def tensor_to_np(self, data):
        return data.detach().to('cpu').numpy()



