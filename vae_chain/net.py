# variational autoencoder の親クラスを定義

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable, optimizers, initializer,cuda
import os
import matplotlib.pyplot as plt
import cupy as cp

log2pi = float(np.log(2*np.pi))

# Xavier initializer : used in class VAE
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



class Net():
    def __init__(self, input_shape, hidden, act_func=F.tanh, out_func=F.sigmoid ,use_BN=False, init_method='xavier', folder='./model', is_gauss_dist=False):

        activations = {
                "sigmoid"   : F.sigmoid,    \
                "tanh"      : F.tanh,       \
                "softplus"  : F.softplus,   \
                "relu"      : F.relu,       \
                "leaky"     : F.leaky_relu, \
                "elu"       : F.elu,        \
                "identity"  : lambda x:x    \
        }
        if isinstance(act_func, str): # 文字列で指定されたとき関数に変換
            if act_func in activations.keys():
                act_func = activations[act_func]
            else:
                print('arg act_func is ', act_func, '. This value cannot be selected. This model uses identity function as activation function.')
                act_func = lambda x:x
        if isinstance(out_func, str): # 文字列で指定されたとき関数に変換
            if out_func in activations.keys():
                out_func = activations[out_func]
            else:
                print('arg out_func is ', out_func, '. This value cannot be selected. This model uses identity function as activation function.')
                out_func = lambda x:x
        if out_func != F.sigmoid:
            print('out_func should be sigmoid')

                
        if isinstance(init_method, str): # 文字列で指定されたとき関数に変換
            inits = {
                    "xavier"    : 'xavier',    \
                    "henormal"  : chainer.initializers.HeNormal \
            }
            if init_method in inits.keys():
                init_method = inits[init_method]
            else:
                print('init_method {} is not exist. This model uses xavier initializer.'.format(init_method))
                init_method = 'xavier'

        if not isinstance(hidden, list): # リストじゃなかったらリストに変換
            hidden = [hidden]
        hidden = [input_shape] + hidden
        print('layer' + str(hidden))

        # model保存用のパス
        self.save_dir  = os.path.join(folder, '{}'.format(hidden))
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.save_dir+'/weight_plot', exist_ok=True)

        self.weight_num = len(hidden)
        self.is_gauss_dist = is_gauss_dist
        self.set_model(hidden, act_func, out_func, use_BN, init_method, is_gauss_dist)
        self.set_optimizer(learning_rate=0.001, gradient_clipping=1.0)

        
    # 多分使う機会はほとんどない
    def __call__(self, x):
        mu,var = self.encode(x)
        z = F.gaussian(mu, var)
        d_out = self.decode(z)
        return z, d_out
        
    # 学習モード
    def train(self,train,epoch,batch,C=1.0,k=1, gpu_num=0, valid=None, is_plot_weight=False):
        min_mse = 1000000

        if gpu_num > -1:
            self.model_to(gpu_num)


        if valid is None:
            print('epoch\tloss\t\tlatent\t\treconst\t\tMSE')
        else:
            print('epoch\tloss\t\tlatent\t\treconst\t\tMSE\t\tvalid')

        nlen = train.shape[0]
        data = train.copy()

        itr = nlen//batch if nlen % batch == 0 else nlen//batch+1

        loss_list = []
        for ep in range(epoch):
            Loss = Reconst = Latent = 0
            perm = np.random.permutation(nlen)
            try:
                for p in range(0,nlen,batch):
                    # encode
                    encoder_input = cuda.to_gpu(data[perm[p:p+batch]], device=gpu_num) if gpu_num>-1 else data[perm[p:p+batch]].copy()
                    encoder_input = Variable( encoder_input )
                    btc = float(encoder_input.shape[0])
                    mu, ln_var = self.encode(encoder_input)
                    latent_loss = C * self.calc_latent_loss(mu, ln_var)# / btc
                    # decode
                    reconst_loss = 0
                    for l in range(k):
                        decoder_input  = F.gaussian(mu, ln_var)
                        decoder_output = self.decode(decoder_input)
                        reconst_loss  += self.calc_reconst_loss(encoder_input, decoder_output) / (k)#*btc)
                    loss = latent_loss + reconst_loss
                    # back prop
                    self.cleargrads()
                    loss.backward()
                    self.update()
                    # using for print loss
                    Loss    += loss
                    Reconst += reconst_loss
                    Latent  += latent_loss
                Loss = float(Loss.data) / itr
                loss_list.append( Loss )

                # show training progress
                if (ep + 1) % 10 == 0 or ep == 0:
                    Reconst = float( Reconst.data ) / itr
                    Latent  = float(  Latent.data ) / itr
                    perm = np.random.permutation(data.shape[0])[:batch]
                    mse = self.MSE(cp.array(data[perm])).mean()
                    pr = '{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}'.format(ep+1, Loss, Latent, Reconst, float(mse))
                    if valid is not None:
                        vld = self.MSE(cp.array(valid)).mean()
                        pr += '\t{:.6f}'.format(float(vld))
                    print(pr)

                    # check layer weight per 10 epoch.
                    if is_plot_weight:
                        self.plot_weight(ep)
                        # self.plot_latent( cp.array(data[perm]), ep )
                    # save best validation model
                    if (valid is not None) and min_mse > mse:
                        os.makedirs(self.save_dir+'/best_valid', exist_ok=True)
                        self.save_model(path = self.save_dir+'/best_valid')
                        min_mse = mse
            except:
                os.makedirs(self.save_dir+'/error_save', exist_ok=True)
                self.save_model(path = self.save_dir+'/error_save')
                print('any error stop program')
                exit()


        self.model_to(-1)

        ll = np.array(loss_list)
        # モデルの保存および, エラーカーブの出力/保存.
        self.save_model(train_loss=ll)

        print('training end\n')

    # オーバーライド用
    def encode(self, encoder_input):
        raise Exception()

    def decode(self, decoder_input):
        raise Exception()

    def calc_latent_loss(self, mu, ln_var):
        return F.gaussian_kl_divergence(mu, ln_var) / mu.size

    def calc_reconst_loss(self, encoder_input, decoder_output):
        if self.is_gauss_dist:
            dec_mu, dec_var = decoder_output
            m_vae = 0.5* (encoder_input - dec_mu)**2 * F.exp(-dec_var)
            a_vae = 0.5* (log2pi+dec_var)
            # reconst = F.sum(m_vae + a_vae, axis=1)
            reconst = F.sum(m_vae + a_vae)
        else:
            reconst = F.bernoulli_nll(encoder_input, decoder_output)
        return reconst / decoder_output.size

    def cleargrads(self):
        raise Exception()

    def update(self):
        raise Exception()

    def model_to(self, gpu_num):
        raise Exception()


    # モデルのセット. オーバーライド用.
    def set_model(self, hidden, act_func, use_BN, init_method, is_gauss_dist):
        raise Exception()

    def set_optimizer(self, learning_rate=0.001, gradient_momentum=0.9, weight_decay=None, gradient_clipping=None):
        raise Exception()

    # 各レイヤーの重みをプロット. 重み更新が機能してるか確認.
    def plot_weight(self, epoch):
        raise Exception()

    def plot_latent(self, data, epoch):
        m,v = self.encode(data)
        plot = F.gaussian(m, v)
        plot = cp.asnumpy(plot.data)
        plt.scatter( plot[:,0], plot[:,1] )
        plt.savefig(self.save_dir + '/weight_plot/latent_{}.png'.format(epoch+1))
        plt.clf()
        plt.close()

    # modelの保存. trainメソッドの最後に呼び出される. ついでにエラーカーブも保存.
    def save_model(self, path=None, train_loss=None):
        raise Exception()

    # modelのロード. 学習済みモデルを使用したい場合に呼び出す.
    def load_model(self, path=None):
        raise Exception()


    # 入力データの 潜在特徴, 再構成データ, 誤差 を返す.
    def reconst(self, data, unregular=False):
        if data.ndim == 1:
            data = data.reshape(1,-1)

        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            e_mu,e_var = self.encode(Variable(data))
            feat = F.gaussian(e_mu, e_var).data
            d_out = self.decode(Variable(feat))

        if self.is_gauss_dist:
            rec = d_out[0].data
            if unregular: # 非正則化項のやつ
                d_mu, d_var = d_out
                # D_VAE = F.gaussian_kl_divergence(e_mu, e_var)
                # A_VAE = 0.5* ( np.log(2*np.pi) + d_var )
                M_VAE = 0.5* ( data-d_mu )**2 * F.exp(-d_var)
                return feat, rec, M_VAE.data
        else:
            # bernoulli_nll内でかけられるsigmoidに帳尻を合わす
            rec = F.sigmoid(d_out).data
            if unregular:
                # bernoulli verでの非正則化項はおそらくこうなる. 間違っているかも.
                p = rec
                v = p*(1-p) # ベルヌーイ分布の分散の式
                M_VAE = 0.5* ( data-p )**2 / v
                return feat, rec, M_VAE
            

        mse = np.mean( (rec-data)**2, axis=1 )

        # lat_loss = F.gaussian_kl_divergence(e_mu, e_var)
        # rec_loss = F.bernoulli_nll( Variable(data), d_out )
        # vae_err = (lat_loss+rec_loss).data

        return feat, rec, mse


    # 再構成誤差を二乗平均で計算. 主に学習時の確認用.
    def MSE(self, data):
        if data.ndim == 1:
            data = data.reshape(1,-1)
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            if isinstance(data, Variable):
                e, d_out = self(data)
            else:
                e, d_out = self(Variable(data))

        if self.is_gauss_dist:
            rec = d_out[0].data
        else:
            rec = F.sigmoid(d_out).data
        return np.mean( (rec-data)**2, axis=1 )


    # 潜在特徴を入力したときのデコーダの出力を返す
    def featuremap_to_image(self, feat):
        if feat.ndim == 1:
            feat = feat.reshape(1,-1)
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            d_out = self.decode(Variable(feat))
        if self.is_gauss_dist:
            output = d_out[0].data
        else:
            output = F.sigmoid(d_out).data
        return output



