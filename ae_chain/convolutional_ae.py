# 畳み込みのVAE実装
# Linearレイヤーのみの実装と比べて次元圧縮サイズの設定が困難なので, エンコードの出力サイズのみ指定可能にする

import numpy as np
from functools import reduce
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable, optimizers, initializer,cuda
import os
import matplotlib.pyplot as plt
import cupy as cp

#from net import Net, Xavier

# train_shape は学習データtrain の train.shape を渡す想定
class Encoder_Decoder(chainer.Chain):
    def __init__(self, train_shape, hidden, enc_func=F.relu, dec_func=F.relu, use_BN=False, init_method='xavier'):
        super(Encoder_Decoder, self).__init__()
        self.use_BN = use_BN
        self.act_func = enc_func
        self.out_func = dec_func

        self.makeLayers( train_shape, hidden, init_method)


    def makeLayers(self, input_shape, hidden, init_method):
        # input_shape[0]...学習データ数
        #            [1]...チャンネル数
        #            [2]...画像で言うと縦 (波形データならここまで)
        #            [3]...画像で言うと横
        input_dim = len(input_shape) - 2
        input_channel = input_shape[1]
        dec_shape = [input_dim, 64] + [ i//8-3 for i in input_shape[2:]]
        # ↑ ksize2, stride2で縦横半分(切り捨て), ksize5,stride2で縦横半分(切り捨て)-2 なので, conv3の後のサイズは内包表記のリストのようになる

        #init_func = Xavier() if init_method=='xavier' else init_method()
        init_func = Xavier if init_method=='xavier' else init_method
        self.add_link('enc_conv1', L.ConvolutionND(input_dim, input_channel, 32, ksize=3))
        self.add_link('enc_conv2', L.ConvolutionND(input_dim, 32, 64, ksize=3))
        self.add_link('enc_conv3', L.ConvolutionND(input_dim, 64, 64, ksize=3))
        if self.use_BN:
            self.add_link('enc_conv1_BN', L.BatchNormalization(32))
            self.add_link('enc_conv2_BN', L.BatchNormalization(64))
            self.add_link('enc_conv3_BN', L.BatchNormalization(64))

        self.add_link('enc_lin1', L.Linear(None, 100))
        self.add_link('enc_lin2', L.Linear(None, 2))

        # self.add_link('dec0_lin', L.Linear(2, reduce(lambda a,b:a*b, dec_shape), initialW=init_func))
        self.add_link('dec0_lin', L.Linear(2, 100))
        self.add_link('dec1_lin', L.Linear(100, 123*64))
        self.add_link('dec_conv1', L.DeconvolutionND(input_dim, None, 64, ksize=3))
        self.add_link('dec_conv2', L.DeconvolutionND(input_dim, 64, 32, ksize=3))
        self.add_link('dec_conv3', L.DeconvolutionND(input_dim, 32, input_channel, ksize=3))
        if self.use_BN:
            self.add_link('dec_conv1_BN', L.BatchNormalization(64))
            self.add_link('dec_conv2_BN', L.BatchNormalization(32))

        self.add_link('dec_out', L.DeconvolutionND(input_dim, input_channel, input_channel, ksize=1, stride=1))


    def __call__(self, x):
        e = self.encode(x)
        d_out = self.decode(e)
        return e,d_out

    def encode(self, x):
        e = x
        for layer in self.children():
            if 'enc' in layer.name:
                #print(layer.name, e.data.shape)
                e = layer(e)
                #if 'enc_conv1' in layer.name:
                #    e = F.sigmoid(e)
                #elif 'enc_lin2' in layer.name:
                #    e = F.tanh(e)
                if False:
                    pass
                else:
                    e = self.act_func(e) if ('BN' in layer.name or not self.use_BN) else e
        return e

    def decode(self, z):
        d = z
        for layer in self.children():
            if 'dec' in layer.name:
                #print(layer.name,  d.data.shape)
                #print(layer.name, d[0:2,0:2])
                if 'out' in layer.name: # 出力レイヤー
                    break
                d = layer(d)
                d = self.act_func(d) if ('BN' in layer.name or not self.use_BN) else d
                if 'dec1_lin' in layer.name:
                    d = F.reshape(d,(-1, 64, 123))


        out = self.dec_out(d)
        return F.sigmoid(out)


class convolutionalAE():

    def __init__(self, input_shape, hidden, enc_func=F.sigmoid, dec_func=F.sigmoid ,use_BN=False, init_method='henormal', folder='./model', rho=None, s=None):
        self.ae_method = None
        if (rho is not None) and (s is not None):
            self.ae_method = 'sparse'
            self.rho = rho
            self.s = s

        def swish(x):
            beta = np.ones(x.shape[1], dtype=np.float32) if isinstance(x.data ,np.ndarray) else cp.ones(x.shape[1], dtype=cp.float32)
            return F.swish(x, beta)

        activations = {
                "sigmoid"   : F.sigmoid,    \
                "tanh"      : F.tanh,       \
                "softplus"  : F.softplus,   \
                "relu"      : F.relu,       \
                "leaky"     : F.leaky_relu, \
                "elu"       : F.elu,        \
                "swish"     : swish,        \
                "identity"  : lambda x:x    \
        }
        if isinstance(enc_func, str): # 文字列で指定されたとき関数に変換
            if enc_func in activations.keys():
                enc_func = activations[enc_func]
            else:
                print('arg enc_func is ', enc_func, '. This value is not exist. This model uses identity function as activation function.')
                enc_func = lambda x:x
        if isinstance(dec_func, str): # 文字列で指定されたとき関数に変換
            if dec_func in activations.keys():
                dec_func = activations[dec_func]
            else:
                print('arg dec_func is ', dec_func, '. This value is not exist. This model uses identity function as activation function of output layer.')
                dec_func = lambda x:x

                
        if isinstance(init_method, str): # 文字列で指定されたとき関数に変換
            inits = {
                    "xavier"    : 'xavier',    \
                    "henormal"  : chainer.initializers.HeNormal \
            }
            if init_method in inits.keys():
                init_method = inits[init_method]
            else:
                print('init_method {} is not exist. This model uses HeNormal initializer.'.format(init_method))
                init_method = inits['henormal']


        # model保存用のパス
        self.save_dir  = folder
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.save_dir+'/weight_plot', exist_ok=True)
        self.model = Encoder_Decoder(input_shape ,hidden, enc_func=enc_func, dec_func=dec_func, use_BN=use_BN, init_method=init_method)
        self.opt = optimizers.Adam()
        # self.opt = optimizers.MomentumSGD()
        self.opt.setup(self.model)

        
        
    # 多分使う機会はほとんどない
    def __call__(self, x):
        e = self.encode(x)
        d = self.decode(e)
        return e, d


    # 学習モード
    def train(self, train, epoch, batch, gpu_num=0, valid=None, is_plot_weight=False):
        min_mse = np.finfo(np.float32).max # float32の最大値

        if gpu_num > -1:
            self.model_to(gpu_num)


        if valid is None:
            print('epoch\tloss\t\tMSE')
        else:
            print('epoch\tloss\t\tMSE\t\tvalid')

        nlen = train.shape[0]
        data = train.copy()

        itr = nlen//batch if nlen % batch == 0 else nlen//batch+1

        loss_list = []
        for ep in range(epoch):
            Loss = 0
            perm = np.random.permutation(nlen)
            for p in range(0,nlen,batch):
                # encode
                encoder_input = cuda.to_gpu(data[perm[p:p+batch]], device=gpu_num) if gpu_num>-1 else data[perm[p:p+batch]].copy()
                encoder_input = Variable( encoder_input )
                btc = float(encoder_input.shape[0])
                decoder_input = self.encode(encoder_input)

                # decode
                decoder_output = self.decode(decoder_input)
                #print(decoder_input)
                #print(decoder_output)
                #exit()
                loss = self.calc_loss(encoder_input, decoder_output)

                # back prop
                self.cleargrads()
                loss.backward()
                self.update()
                # using for print loss
                Loss    += loss
            Loss = float(Loss.data) / itr
            loss_list.append( Loss )

            # show training progress
            if (ep + 1) % 10 == 0 or ep == 0:
                perm = np.random.permutation(data.shape[0])[:batch]
                mse = self.MSE(cp.array(data[perm])).mean()
                pr = '{}\t{:.6f}\t{:.6f}'.format(ep+1, Loss, float(mse))
                if valid is not None:
                    vld = self.MSE(cp.array(valid)).mean()
                    pr += '\t{:.6f}'.format(float(vld))
                print(pr)

                # check layer weight per 10 epoch.
                if is_plot_weight:
                    self.plot_weight(ep)
                # save best validation model
                if (valid is not None) and min_mse > mse:
                    os.makedirs(self.save_dir+'/best_valid', exist_ok=True)
                    self.save_model(path = self.save_dir+'/best_valid')
                    min_mse = mse

        self.model_to(-1)

        ll = np.array(loss_list)
        # モデルの保存および, エラーカーブの出力/保存.
        self.save_model(train_loss=ll)

        print('training end\n')



    def encode(self, encoder_input):
        return self.model.encode(encoder_input)

    def decode(self, decoder_input):
        return self.model.decode(decoder_input)


    def calc_loss(self, encoder_input, decoder_output):
        if self.ae_method == 'sparse':
            loss = F.mean_squared_error(decoder_output, encoder_input)
            if 0 < self.s:
                rho_hat = F.sum(decoder_output, axis=0) / decoder_output.shape[0]
                kld = F.sum(self.rho * F.log(self.rho / rho_hat) + \
                    (1 - self.rho) * F.log((1 - self.rho) / (1 - rho_hat)))
                loss += self.s * kld
        else:
            loss = F.mean_squared_error(decoder_output, encoder_input)
        return loss


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
            plt.legend(loc='upper right')
        plt.tight_layout()
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

    # 再構成データと入力データのMSEを返す. MSEはデータ長Nのベクトル(Nは入力データの数)
    def MSE(self, data):
        if data.ndim == 1:
            data = data.reshape(1,-1)
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            if isinstance(data, Variable):
                feat, recon = self(data)
            else:
                feat, recon = self(Variable(data))
        recon = recon.data
        return np.mean( (recon-data)**2, axis=1 )


    # 入力データの 潜在特徴, 再構成データ, 誤差 を返す.
    def reconst(self, data):
        if data.ndim == 1:
            data = data.reshape(1,-1)

        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            if isinstance(data, Variable):
                feat, recon = self(data)
            else:
                feat, recon = self(Variable(data))
        feat = feat.data
        recon = recon.data
        mse = np.mean( (recon-data)**2, axis=1 )
        return feat, recon, mse



