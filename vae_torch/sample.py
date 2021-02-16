# VAEのサンプルコード, MNIST使用

import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    import sys
    
    # コマンドライン引数を読み込み
    # 引数が'-1'なら学習しない
    args = sys.argv
    train_mode = ['train', 'retrain', 'load']
    mode = 0 if len(args) < 2 else int(args[1])
    train_mode = train_mode[mode]
    
    # torchのMNISTの呼び出し方がよくわからんかったのでchainerで代用
    # MNISTデータの読み込み
    import chainer
    train, test = chainer.datasets.get_mnist()
    # データとラベルに分割
    train_data, train_label = train._datasets
    test_data, test_label = test._datasets

    # 標準化
    # tr_std = train_data.std()
    # tr_avg = train_data.mean()
    # train_data = (train_data - tr_avg) / tr_std
    # test_data  = (test_data  - tr_avg) / tr_std

    # 学習条件
    # エポックとミニバッチサイズ
    epoch = 100
    batchsize = 128
    # 隠れ層のユニット数
    hidden = [128,2]

    act_func = 'tanh'    # 活性化関数
    out_func = 'sigmoid' # 出力層の活性化関数 (デコーダの出力層, 無印版は必ずsigmoid)
    use_BN   = True      # Batch Normalization を使うか否か

    fd = './model/torch/'

    # modelのセットアップ
    from variational_autoencoder import VAE
    vae = VAE( int(train_data.shape[1]) ,hidden, act_func=act_func, out_func=out_func ,use_BN=use_BN, folder=fd, device='cuda', is_gauss_dist=True)

    # ed版はこちら. 無印版とは名前が違うだけ.
    # from variational_autoencoder_ed import VAE_ED
    # vae = VAE_ED( int(train_data.shape[1]) ,hidden, act_func=act_func, out_func=out_func ,use_BN=True, folder=fd, device='cuda', is_gauss_dist=True)

    # VAEの学習
    if train_mode == 'train':
        # vae.train(train_data, epoch, batchsize,C=1.0, k=1, valid=None)
        vae.train(train_data, epoch, batchsize,C=1.0, k=1, valid=test_data, is_plot_weight=True)
    if train_mode == 'retrain':
        # もしかしたらoptimizerも保存・ロードしたほうがいいかもしれない
        vae.load_model()
        vae.train(train_data, epoch, batchsize,C=1.0, k=1, valid=None)
    else:
        vae.load_model()

    # 評価モードに切り替え. batch normalizationの無効化などに必須
    vae.model_to_eval()




    # テストデータを絞る, なおこのコードではテストデータを見ていない
    # test_data = test_data[:20]
    # test_label = test_label[:20]

    # 再構成
    feat_train, reconst_train, err_train = vae.reconst(train_data)
    feat_test,  reconst_test,  err_test  = vae.reconst(test_data)

    # 再構成データとオリジナルを1次元で比較
    plt.plot(reconst_train[0])
    plt.plot(train_data[0])
    plt.show()
    
    # 潜在空間の可視化
    col = ['r','g','b','c','m','y','orange','black','gray','violet']
    for i in range(3000):
        plt.scatter(x=feat_train[i,0],y=feat_train[i,1],marker='.',color=col[train_label[i]])
    plt.show()


    # 自作の潜在空間から出力を画像を確認. VAEで調べるとよく見る数字がシームレスに変化するやつ.
    split_num = 20  # 分割数. 20より大きくするのは重いのでやめたほうがいい
    rn = np.linspace(-3,3,split_num)
    x,y = np.meshgrid(rn, rn)
    dt  = np.hstack( [x.reshape(-1,1), y.reshape(-1,1)] )
    # 変換するメソッド
    imgs = vae.featuremap_to_image(dt)

    for i in range(split_num**2):
        plt.subplot(split_num,split_num,i+1) 
        plt.imshow(imgs[i].reshape((28,28)),cmap='gray')
    plt.show()
    



