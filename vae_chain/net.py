# variational autoencoder の親クラスを定義

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable, optimizers, initializer,cuda
import os
import matplotlib.pyplot as plt
import cupy as cp

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



class net():
    def __init__(self, input_shape, hidden, act_func=F.tanh, use_BN=False, folder='./model', init_method='xavier'):

        if isinstance(act_func, str): # 文字列で指定されたとき関数に変換
            activations = {
                    "sigmoid"   : F.sigmoid,    \
                    "tanh"      : F.tanh,       \
                    "softplus"  : F.softplus,   \
                    "relu"      : F.relu,       \
                    "leaky"     : F.leaky_relu, \
                    "elu"       : F.elu         \
            }
            if act_func in activations.keys():
                act_func = activations[act_func]
            else:
                print('arg act_func is ', act_func, '. This value is not exist. This model uses identity function as activation function.')
                act_func = lambda x:x
                
        if isinstance(init_method, str): # 文字列で指定されたとき関数に変換
            inits = {
                    "xavier"    : 'xavier',    \
                    "henormal"  : chainer.initializers.HeNormal() \
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


        self.set_model(hidden, act_func, use_BN, init_method)
        
    # 多分使う機会はほとんどない
    def __call__(self, x):
        raise Exception()
        

    # 学習モード
    def train(self,train,epoch,batch,C=1.0,k=1, gpu_num=0,valid=None, is_plot_weight=False):
        self.set_optimizer(learning_rate=0.001, gradient_clipping=1.0 ,gpu_num=gpu_num)

        if valid is None:
            print('epoch\tloss\t\tlatent\t\treconst\t\tMSE')
        else:
            print('epoch\tloss\t\tlatent\t\treconst\t\tMSE\t\tvalid')

        nlen = train.shape[0]
        data = train.copy()

        itr = nlen // batch if nlen % batch == 0 else nlen//batch+1

        min_mse = 1000000
        loss_list = []
        for ep in range(epoch):
            Loss = Reconst = Latent = 0
            perm = np.random.permutation(nlen)
            for p in range(0,nlen,batch):

                encoder_input = Variable( cuda.to_gpu(data[perm[p:p+batch]], device=gpu_num) )
                btc = float(encoder_input.shape[0])

                mu, ln_var  = self.encode(encoder_input)
                latent_loss = self.calc_latent_loss(mu, ln_var, C, btc)

                reconst_loss = 0
                for l in range(k):
                    decoder_input  = F.gaussian(mu, ln_var)
                    decoder_output = self.decode(decoder_input)

                    reconst_loss += self.calc_reconst_loss(encoder_input, decoder_output, k, btc)
                loss = latent_loss + reconst_loss

                # 逆伝搬 
                self.cleargrads()
                loss.backward()
                self.update()

                # print用
                Loss    += loss
                Reconst += reconst_loss
                Latent  += latent_loss
            Loss = float(Loss.data) / itr
            loss_list.append( Loss )

            if (ep + 1) % 10 == 0 or ep == 0:
                Reconst = float( Reconst.data ) / itr
                Latent  = float(  Latent.data ) / itr
                perm = np.random.permutation(batch)
                mse = self.MSE(cp.array(data[perm])).mean()
                pr = '{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}'.format(ep+1, Loss, Latent, Reconst, float(mse))
                if valid is not None:
                    vld = self.MSE(cp.array(valid)).mean()
                    pr += '\t{:.6f}'.format(float(vld))
                print(pr)

                if is_plot_weight: # 重みの更新履歴を確認したいときに実行. epochごと.
                    self.plot_weight(ep)
                if (valid is not None) and min_mse > mse:
                    os.makedirs(self.save_dir+'/best_valid', exist_ok=True)
                    self.save_model(path = self.save_dir+'/best_valid')
                    min_mse = mse

        self.model.to_cpu()

        ll = np.array(loss_list)
        # モデルの保存および, エラーカーブの出力/保存.
        self.save_model(train_loss=ll)

        print('training end\n')


    # オーバーライド用
    def encode(self, encoder_input):
        raise Exception()

    def decode(self, decoder_input):
        raise Exception()

    def calc_latent_loss(self, mu, ln_var, C=1.0, batch=1.0):
        raise Exception()

    def calc_reconst_loss(self, encoder_input, decoder_output, k=1.0, batch=1.0):
        raise Exception()

    def cleargrads(self):
        raise Exception()

    def update(self):
        raise Exception()


    # モデルのセット. オーバーライド用.
    def set_model(self, hidden, act_func, use_BN, init_method):
        self.model = Encoder_Decoder(hidden, act_func, use_BN, init_method)

    def set_optimizer(self, learning_rate=0.001, gradient_momentum=0.9, weight_decay=None, gradient_clipping=None, gpu_num=0):
        self.model.to_gpu(gpu_num)
        self.opt = optimizers.Adam(alpha=learning_rate, beta1=gradient_momentum)
        # self.opt = optimizers.Adam()
        self.opt.setup(self.model)

        if gradient_clipping is not None:
            self.opt.add_hook(chainer.optimizer.GradientClipping(gradient_clipping))
        if weight_decay is not None:
            self.opt.add_hook(chainer.optimizer.WeightDecay(0.001))


    # 各レイヤーの重みをプロット. 重み更新が機能してるか確認.
    def plot_weight(self, epoch):
        # BN以外のレイヤー数を取得. もうちょっとなんとかなったやろこれ.
        layer_num = (len( list(filter(lambda x: not('BN' in x.name), self.model.children() )) )+1) //2
        
        j = 1
        fig = plt.figure(figsize=(16,8))
        for layer in self.model.children():
            if 'BN' in layer.name:
                continue
            plt.subplot(2,layer_num,j)
            plt.plot(cp.asnumpy(layer.W.data).reshape(-1), label=layer.name)
            plt.legend()
            j+=1
            if 'out' in layer.name:
                j+=1
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

    # 再構成誤差を二乗平均で計算. 主に学習時の確認用.
    def MSE(self, data, use_gpu=False):
        if data.ndim == 1:
            data = data.reshape(1,-1)
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            if isinstance(data, Variable):
                e, d_out = self(data)
            else:
                e, d_out = self(Variable(data))
        rec = F.sigmoid(d_out).data
        return np.mean( (rec-data)**2, axis=1 )

