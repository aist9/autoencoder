
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import chainer
import chainer.functions as F
import chainer.links as L
import chainer.computational_graph as c
from chainer import Chain
from chainer import Variable, optimizers, cuda
from chainer import datasets, iterators
import cupy as cp

# **********************************************
# autoencoder class
# **********************************************
class Autoencoder(Chain):
    def __init__(self, inputs, hidden, enc_func, dec_func, use_BN=False, init_method=''):
        super(Autoencoder, self).__init__()
        with self.init_scope():
            self.le = L.Linear(inputs, hidden)
            self.ld = L.Linear(hidden, inputs)
            if use_BN:
                self.be = L.BatchNormalization(hidden)
                self.bd = L.BatchNormalization(inputs)

        # 活性化関数の指定
        self.fe = enc_func
        self.fd = dec_func

        self.use_BN = use_BN


    # Forward
    # hidden_out=Trueにすると隠れ層出力もreturn
    def __call__(self, x, hidden_out=False):
        h = self.encoder(x)
        y = self.decoder(h)
        if hidden_out == False:
            return y
        else:
            return y, h

    # Encoder
    def encoder(self, x):
        h = self.le(x)
        if self.use_BN:
            h = self.be(h)
        if self.fe is not None:
            h = self.fe(h)
        return h
        
    # Decoder
    def decoder(self, h):
        y = self.ld(h)
        if self.use_BN:
            y = self.bd(y)
        if self.fd is not None:
            y = self.fd(y) 
        return y
    


class AE():
    def __init__(self, input_shape, hidden, enc_func=F.sigmoid, dec_func=F.sigmoid ,use_BN=False, init_method='henormal', folder='./model', rho=None, s=None):

        self.ae_method = None
        if (rho is not None) and (s is not None):
            self.ae_method = 'sparse'
            self.rho = rho
            self.s = s

        activations = {
                "sigmoid"   : F.sigmoid,    \
                "tanh"      : F.tanh,       \
                "softplus"  : F.softplus,   \
                "relu"      : F.relu,       \
                "leaky"     : F.leaky_relu, \
                "elu"       : F.elu,        \
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

        print('layer', str(input_shape), str(hidden))
        if not isinstance(hidden, int):
            print('type error : arg "hidden" must be integer.')
            print('             If you give list, you should use StackedAE class.')
            exit()

        # model保存用のパス
        self.save_dir  = folder
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.save_dir+'/weight_plot', exist_ok=True)
        self.model_name = "{}-{}.npz".format(input_shape, hidden)

        self.model = Autoencoder(input_shape ,hidden, enc_func=enc_func, dec_func=dec_func, use_BN=use_BN, init_method=init_method)
        self.opt = optimizers.Adam()
        self.opt.setup(self.model)

        
    # 多分使う機会はほとんどない
    def __call__(self, x):
        e = self.encode(x)
        d = self.decode(e)
        return e, d
        
    # 学習モード
    def train(self, train, epoch, batch, gpu_num=0, valid=None, is_plot_weight=False):
        min_mse = 1000000

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


    # オーバーライド用
    def encode(self, encoder_input):
        return self.model.encoder(encoder_input)

    def decode(self, decoder_input):
        return self.model.decoder(decoder_input)


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

    def model_to(self, gpu_num):
        if gpu_num == -1:
            self.model.to_cpu()
        else:
            self.model.to_gpu(gpu_num)

    # 各レイヤーの重みをプロット. 重み更新が機能してるか確認.
    def plot_weight(self, epoch):
        fig = plt.figure(figsize=(16,8))
        plt.subplot(121)
        plt.plot(cp.asnumpy(self.model.le.W.data).reshape(-1), label='encoder')
        plt.legend(loc='upper right')
        plt.subplot(122)
        plt.plot(cp.asnumpy(self.model.ld.W.data).reshape(-1), label='decoder')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig( self.save_dir + '/weight_plot/{}_ep_{}.png'.format(self.model_name[:-4], epoch+1) )
        plt.close()

    # modelの保存. trainメソッドの最後に呼び出される. ついでにエラーカーブも保存.
    def save_model(self, path=None, train_loss=None):
        path = self.save_dir if path is None else path
        save_path = os.path.join(path, self.model_name)
        chainer.serializers.save_npz(save_path, self.model)
        
    # modelのロード. 学習済みモデルを使用したい場合に呼び出す.
    def load_model(self, path=None):
        path = self.save_dir if path is None else path
        save_path = os.path.join(path, self.model_name)
        chainer.serializers.load_npz( save_path, self.model )

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


class StackedAE(AE):
    def __init__(self, input_shape, hidden, enc_func=[F.sigmoid], dec_func=[F.sigmoid] ,use_BN=False, init_method='henormal', folder='./model', rho=None, s=None):

        self.ae_method = None
        if (rho is not None) and (s is not None):
            self.ae_method = 'sparse'
            self.rho = rho[0] if isinstance(rho, list) else rho
            self.s   =   s[0] if isinstance(  s, list) else s

        if not isinstance(hidden, list):
            hidden = [hidden]
        hidden = [input_shape] + hidden

        if len(hidden) == 2:
            print('len(hidden)==1. We recommend use AE class.')

        self.AE_list = []
        for i in range(len(hidden)-1):
            enc = enc_func[i] if isinstance(enc_func, list) else enc_func
            dec = dec_func[i] if isinstance(dec_func, list) else dec_func
            inm = init_method[i] if isinstance(init_method, list) else init_method
            r  = rho[i] if isinstance(rho, list) else rho
            s_ = s[i]   if isinstance(s, list)   else s

            ae = AE(hidden[i], hidden[i+1], enc_func=enc, dec_func=dec, use_BN=use_BN, init_method=inm, folder=folder, rho=r, s=s_)
            self.AE_list.append(ae)

        self.save_dir = folder


    # 各層を学習 -> 最後に全層を通して学習(is_last_train=True時). trainメソッドをそのまま継承したかったのでメソッド名を変更.
    def stacked_train(self, train, epoch, batch, gpu_num=0, valid=None, is_plot_weight=False, is_last_train=True):

        data = train.copy()
        v_data = valid
        for i in range(len(self.AE_list)):
            print('training layer', i+1)
            self.AE_list[i].train(data, epoch, batch, gpu_num, v_data, is_plot_weight)
            data  = self.AE_list[i].encode(data).data
            v_data = None if v_data is None else self.AE_list[i].encode(v_data).data

        # メインコードでtrain呼べばいいだけなのでこの分岐は不要かも
        if is_last_train:
            print('training all at once')
            self.train(train, epoch, batch, gpu_num, valid, is_plot_weight)

    # 継承そのまま
    # def train(self, train, epoch, batch, gpu_num=0, valid=None, is_plot_weight=False):


    def encode(self, encoder_input):
        e = encoder_input
        for ae in self.AE_list:
            e = ae.encode(e)
        return e

    def decode(self, decoder_input):
        d = decoder_input
        for ae in self.AE_list[::-1]:
            d = ae.decode(d)
        return d

    def cleargrads(self):
        for ae in self.AE_list:
            ae.cleargrads()

    def update(self):
        for ae in self.AE_list:
            ae.update()

    def model_to(self, gpu_num):
        for ae in self.AE_list:
            ae.model_to(gpu_num)

    # 各レイヤーの重みをプロット. 重み更新が機能してるか確認.
    def plot_weight(self, epoch):
        fig = plt.figure(figsize=(16,8))
        l_num = len(self.AE_list)
        for i in range(l_num):
            plt.subplot(2,l_num,1+i)
            plt.plot( cp.asnumpy(self.AE_list[i].model.le.W.data).reshape(-1), label='enc_'+str(i))
            plt.legend(loc='upper right')
            plt.subplot(2,l_num,1+i+l_num)
            plt.plot( cp.asnumpy(self.AE_list[i].model.ld.W.data).reshape(-1), label='dec_'+str(i))
            plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig( self.save_dir + '/weight_plot/ep_{}.png'.format(epoch+1) )
        plt.close()

    # modelの保存. trainメソッドの最後に呼び出される. ついでにエラーカーブも保存.
    def save_model(self, path=None, train_loss=None):
        path = self.save_dir if path is None else path
        for ae in self.AE_list:
            ae.save_model(path, train_loss)
        
    # modelのロード. 学習済みモデルを使用したい場合に呼び出す.
    def load_model(self, path=None):
        path = self.save_dir if path is None else path
        for ae in self.AE_list:
            ae.load_model(path)

    # 継承そのまま
    # def MSE(self, data):

    # 継承そのまま
    # def reconst(self, data):



def main():
 
    # -------------------------------------
    # 設定
    # -------------------------------------

    # コマンドライン引数を読み込み
    # 引数が'-1'なら学習しない
    args = sys.argv
    train_mode = True 
    if 2 <= len(args):
        if args[1] == '-1':
            train_mode = False
    
    # 出力先のフォルダを生成
    save_dir = './model'
    os.makedirs(save_dir, exist_ok=True)

    # -------------------------------------
    # データの準備
    # -------------------------------------

    # 指定した数字データを抜くための変数
    number = 0

    # MNISTデータの読み込み
    train, test = chainer.datasets.get_mnist()

    # データとラベルを取得
    train_data, train_label = train._datasets
    test_data, test_label = test._datasets
    
    # 学習データ: 特定の番号のみ抽出したデータを用いる
    # train_data = train_data[train_label == number]

    # テストデータ: ト特定の番号と適当なデータを合計で10個取得
    # test_data_n  = test_data[test_label == number]
    # test_label_n = test_label[test_label == number]
    # test_data  = np.concatenate((test_data_n[0:5],  test_data[0:5]))
    # test_label = np.concatenate((test_label_n[0:5], test_label[0:5]))

    # -------------------------------------
    # 学習の準備
    # -------------------------------------

    # エポック
    epoch = 100
    # ミニバッチサイズ
    batchsize = 64
    # 隠れ層のユニット数
    # hidden = 64
    hidden = [128,2]

    # -------------------------------------
    # AutoEncoderの学習
    # -------------------------------------

    # コマンドライン引数が'-1'の場合学習しない
    # ae = AE( int(train_data.shape[1]) ,hidden, enc_func='sigmoid', dec_func='sigmoid', use_BN=True, folder=save_dir)
    ae = StackedAE( int(train_data.shape[1]) ,hidden, enc_func='sigmoid', dec_func='sigmoid', use_BN=True, folder=save_dir)
    if train_mode:
        # Autoencoderの学習
        # ae.train(train_data, epoch, batchsize, gpu_num=0, valid=None, is_plot_weight=True)
        # StackedAE限定の学習
        ae.stacked_train(train_data, epoch, batchsize, gpu_num=0, valid=None, is_plot_weight=True, is_last_train=False)
    else:
        # 保存したモデルから読み込み
        ae.load_model()

    # -------------------------------------
    # 再構成
    # -------------------------------------
    # 再構成を実行
    feat_train, reconst_train, err_train = ae.reconst(train_data)
    # feat_test,  reconst_test,  err_test  = ae.reconst(test_data)

    # -------------------------------------
    # 可視化
    # -------------------------------------
    plt.plot(reconst_train[0])
    plt.plot(train_data[0])
    plt.show()
    # exit()

    split_num = 20
    plot_image = np.ones( (split_num*29, split_num*29) ).astype(np.float)
    for i in range(split_num):
        for j in range(split_num):
            plot_image[i*28+i:-~i*28+i,j*28+j:-~j*28+j] = reconst_train[i*split_num+j].reshape(28,28)
    
    plt.imshow(plot_image,cmap='gray', vmax=plot_image.max(), vmin=plot_image.min())
    plt.show()



    col = ['r','g','b','c','m','y','orange','black','gray','violet']
    for i in range(3000):
        plt.scatter(x=feat_train[i,0],y=feat_train[i,1],marker='.',color=col[train_label[i]])
    plt.show()

if __name__ == '__main__':
    main()

