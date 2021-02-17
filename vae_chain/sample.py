import numpy as np
import matplotlib.pyplot as plt

from variational_autoencoder import VAE

if __name__ == '__main__':
    import sys
    
    # コマンドライン引数を読み込み
    # 引数が'-1'なら学習しない
    args = sys.argv
    train_mode = ['train', 'retrain', 'load']
    mode = 0 if len(args) < 2 else int(args[1])
    train_mode = train_mode[mode]
    
    import chainer
    # MNISTデータの読み込み
    train, test = chainer.datasets.get_mnist()
    # データとラベルに分割
    train_data, train_label = train._datasets
    test_data, test_label = test._datasets

    tr_std = train_data.std()
    tr_avg = train_data.mean()
    train_data = (train_data - tr_avg) / tr_std
    test_data  = (test_data  - tr_avg) / tr_std

    test_data = test_data[:20]
    test_label = test_label[:20]

    # 学習の条件
    # エポックとミニバッチサイズ
    epoch = 100
    batchsize = 32 *4
    # 隠れ層のユニット数
    hidden = [128,2]
    # hidden = [600,2]

    act = 'tanh'
    out_func = ['identity', 'sigmoid'][1]
    igd = True
    fd = './model/vae/'

    # modelのセットアップ
    vae = VAE( int(train_data.shape[1]) ,hidden, act_func=act, out_func=out_func, use_BN=True, folder=fd, is_gauss_dist=igd)
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
    for i in range(3000):
        plt.scatter(x=feat_train[i,0],y=feat_train[i,1],marker='.',color=col[train_label[i]])
    plt.show()


    # 自作の潜在空間から出力を画像を確認. VAEで調べるとよく見る数字がシームレスに変化するやつ.
    split_num = 20  # 分割数. 出力画像のサイズは split_num*29 × split_num*29. (MNISTの縦横28+分割線1)
    rn = np.linspace(-3,3,split_num)
    x,y = np.meshgrid(rn, rn)
    dt  = np.hstack( [x.reshape(-1,1), y.reshape(-1,1)] ).astype(np.float32)
    # 変換するメソッド
    imgs = vae.featuremap_to_image(dt)
    plot_image = np.ones( (split_num*29, split_num*29) ).astype(np.float)
    for i in range(split_num):
        for j in range(split_num):
            plot_image[i*28+i:-~i*28+i,j*28+j:-~j*28+j] = imgs[i*split_num+j].reshape(28,28)
    
    plt.imshow(plot_image,cmap='gray', vmax=plot_image.max(), vmin=plot_image.min())
    plt.show()
    

