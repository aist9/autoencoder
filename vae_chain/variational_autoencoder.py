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


#Variational AutoEncoder model
class Encoder_Decoder(chainer.Chain):
    def __init__(self, layers, act_func=F.tanh, use_BN=False, init_method='xavier'):
        super(Encoder_Decoder, self).__init__()
        self.use_BN = use_BN
        self.act_func = act_func
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
            self.add_link('dec_out' ,L.Linear(hidden[1],hidden[0],  initialW=Xavier(hidden[1] , hidden[0] )))
        else:
            self.add_link('enc_mu' ,L.Linear(hidden[-2],hidden[-1], initialW=init_method))
            self.add_link('enc_var',L.Linear(hidden[-2],hidden[-1], initialW=init_method))
            self.add_link('dec_out' ,L.Linear(hidden[1],hidden[0],  initialW=init_method))

        
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
                    out = layer(d)
                    break
                d = layer(d)
                d = self.act_func(d) if ('BN' in layer.name or not self.use_BN) else d

        return out


class VAE():
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
                init_method = 'xavier'
                print('init_method is xavier initializer')


        if not isinstance(hidden, list): # リストじゃなかったらリストに変換
            hidden = [hidden]
        hidden = [input_shape] + hidden
        print('layer' + str(hidden))

        # model保存用のパス
        self.save_dir  = os.path.join(folder, '{}'.format(hidden))
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.save_dir+'/weight_plot', exist_ok=True)


        self.model = Encoder_Decoder(hidden, act_func, use_BN, init_method)
        
    # 多分使う機会はほとんどない
    def __call__(self, x):
        return self.model(x)


    # 学習モード
    def train(self,train,epoch,batch,C=1.0,k=1, gpu_num=0,valid=None, is_plot_weight=False):
        min_mse = 1000000
        self.set_optimizer(learning_rate=0.001, gradient_clipping=1.0 ,gpu_num=gpu_num)

        if valid is None:
            print('epoch\tloss\t\tlatent\t\treconst\t\tMSE')
        else:
            print('epoch\tloss\t\tlatent\t\treconst\t\tMSE\t\tvalid')

        nlen = train.shape[0]
        data = train.copy()

        itr = nlen // batch if nlen % batch == 0 else nlen//batch+1

        loss_list = []
        for ep in range(epoch):
            Loss = Reconst = Latent = 0
            perm = np.random.permutation(nlen)
            for p in range(0,nlen,batch):

                encoder_input = Variable( cuda.to_gpu(data[perm[p:p+batch]], device=gpu_num) )
                btc = float(encoder_input.shape[0])

                mu, ln_var = self.model.encode(encoder_input)
                latent_loss = C * F.gaussian_kl_divergence(mu, ln_var) / btc

                reconst_loss = 0
                for l in range(k):
                    decoder_input  = F.gaussian(mu, ln_var)
                    decoder_output = self.model.decode(decoder_input)

                    reconst_loss += F.bernoulli_nll(encoder_input, decoder_output) / (k*btc)
                loss = latent_loss + reconst_loss

                # 逆伝搬 
                self.model.cleargrads()
                loss.backward()
                self.opt.update()

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

    # 入力データの 潜在特徴, 再構成データ, 誤差 を返す.
    def reconst(self, data, unregular=False):
        if data.ndim == 1:
            data = data.reshape(1,-1)

        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            e_mu,e_var = self.model.encode(Variable(data))
            feat = F.gaussian(e_mu, e_var).data
            d_out = self.model.decode(feat)

        rec = F.sigmoid(d_out).data
        mse = np.mean( (rec-data)**2, axis=1 )

        # lat_loss = F.gaussian_kl_divergence(e_mu, e_var)
        # rec_loss = F.bernoulli_nll( Variable(data), d_out )
        # vae_err = (lat_loss+rec_loss).data

        return feat, rec, mse

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

    # 潜在特徴を入力したときのデコーダの出力を返す
    def featuremap_to_image(self, feat):
        if feat.ndim == 1:
            feat = feat.reshape(1,-1)

        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            d_out = self.model.decode(Variable(feat))
        output = F.sigmoid(d_out).data
        return output


if __name__ == '__main__':
    import sys
    
    # コマンドライン引数を読み込み
    # 引数が'-1'なら学習しない
    args = sys.argv
    train_mode = ['train', 'retrain', 'load']
    mode = 0 if len(args) < 2 else int(args[1])
    train_mode = train_mode[mode]
    
    # MNISTデータの読み込み
    train, test = chainer.datasets.get_mnist()
    # データとラベルに分割
    train_data, train_label = train._datasets
    test_data, test_label = test._datasets

    # tr_std = train_data.std()
    # tr_avg = train_data.mean()
    # train_data = (train_data - tr_avg) / tr_std
    # test_data  = (test_data  - tr_avg) / tr_std

    test_data = test_data[:20]
    test_label = test_label[:20]

    # 学習の条件
    # エポックとミニバッチサイズ
    epoch = 300
    batchsize = 32
    # 隠れ層のユニット数
    hidden = [128,2]
    # hidden = [600,2]

    act = 'tanh'
    fd = './model/vae_gauss/'

    # modelのセットアップ
    vae = VAE( int(train_data.shape[1]) ,hidden, act_func=act, use_BN=True, folder=fd)
    # VAEの学習
    if train_mode == 'train':
        vae.train(train_data, epoch, batchsize, k=1, gpu_num=0, valid=None, is_plot_weight=True)
    if train_mode == 'retrain':
        vae.load_model()
        vae.train(train_data, epoch, batchsize, k=1, gpu_num=0, valid=None, is_plot_weight=True)
    else:
        vae.load_model()

    # 再構成
    feat_train, reconst_train, err_train = vae.reconst(train_data)
    feat_test,  reconst_test,  err_test  = vae.reconst(test_data)


    plt.plot(reconst_train[0])
    plt.plot(train_data[0])
    plt.show()
    # exit()


    col = ['r','g','b','c','m','y','orange','black','gray','violet']
    c_list = []
    for i in range(train_label.shape[0]):
        c_list.append(col[train_label[i]])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(3000):
        ax.scatter(x=feat_train[i,0],y=feat_train[i,1],marker='.',color=c_list[i])
    plt.show()


    rn = np.arange(-3,3,0.3)
    dt = []
    for i in range(20):
        for j in range(20):
            dt.append( np.array( [rn[i],rn[j]],np.float32) )
    dt=np.asarray(dt)

    rc = vae.featuremap_to_image(dt)

    fig = plt.figure()
    for i in range(0,400):
        plots = rc[i]
        # plots = (plots - plots.min()) / ( plots.max() - plots.min() )
        ax1 = fig.add_subplot(20,20,i+1) 
        ax1.imshow(plots.reshape((28,28)),cmap='gray')
        #cv2.waitKey(1)
    plt.show()
    



