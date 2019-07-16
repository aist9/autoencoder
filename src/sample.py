
import sys
import numpy as np
import matplotlib.pyplot as plt
# Chainer
import chainer
from chainer import serializers
# autoencoder.py
import autoencoder
from autoencoder import train_ae, Reconst, Autoencoder

# MNISTを使用して特定の手書き数字データを学習するAutoEncoder
if __name__ == '__main__':
    
    # コマンドライン引数を読み込み
    # 引数が'-1'なら学習しない
    args = sys.argv
    train_mode = True 
    if 2 <= len(args):
        if args[1] == '-1':
            train_mode = False

    # 学習する手書き数字の数値
    number = 5

    # 学習の条件
    # エポックとミニバッチサイズ
    epoch = 200
    batchsize = 32
    # 隠れ層のユニット数
    hidden = 100

    # MNISTデータの読み込み
    train, test = chainer.datasets.get_mnist()
    
    # データとラベルに切り分け(ラベルは不要)
    # 指定したラベルで抜き取り
    train_data, train_label = train._datasets
    train_data = train_data[train_label == number]

    test_data, test_label = test._datasets
    test_data_n  = test_data[test_label == number]
    test_label_n = test_label[test_label == number]
    test_data  = np.concatenate((test_data_n[0:5],  test_data[0:5]))
    test_label = np.concatenate((test_label_n[0:5], test_label[0:5]))

    # AutoEncoderの学習
    # コマンドライン引数が'-1'の場合学習しない
    if train_mode:
        # 学習＋モデルの保存
        model = train_ae(train_data, train_data.shape[1], hidden, epoch, batchsize)
        serializers.save_npz("../output/mymodel.npz", model)

    # 保存したモデルから読み込み
    model = Autoencoder(784, hidden)
    serializers.load_npz("../output/mymodel.npz", model)

    # 再構成を行う
    # AutoEncoderの再構成を行うクラスを定義
    ar = Reconst([model])
    feat_train, reconst_train, err_train = ar(train_data)
    feat_test,  reconst_test,  err_test  = ar(test_data)

    # plt.figure()
    # plt.imshow(test_datas[13].reshape((28,28)),cmap='gray')
    # plt.title('dat1')

    # plt.show()
    # plt.figure()
    # plt.imshow(r[13].reshape((28,28)),cmap='gray')
    # plt.title('dat2')
    # plt.show()

    mn, std, th = ar.err2threshold(train_err, 2)
    print("th: ", th)

    #print(labels[0:9])
    print("label: ",test_label)

    result = ar.judgment(test_err, th)
    print("result: ", result)
    print("errors: ", test_err)

