# 60-300 100-300 200-300

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Chainer
import chainer
from chainer import serializers
# autoencoder.py
import variational_autoencoder
from variational_autoencoder import train_stacked, Reconst, VAE
import cv2

# MNISTを使用して特定の手書き数字データを学習するAutoEncoder
if __name__ == '__main__':
    
    # コマンドライン引数を読み込み
    # 引数が'-1'なら学習しない
    args = sys.argv
    train_mode = True 
    if 2 <= len(args):
        if args[1] == '-1':
            train_mode = False
    
    # 出力先のフォルダを生成
    save_dir = '../output/'
    os.makedirs(save_dir, exist_ok = True)
    # 作成するモデルの保存先+名前
    save_model_name = os.path.join(save_dir, 'mymodel.npz')

    # MNISTデータの読み込み
    train, test = chainer.datasets.get_mnist()
    
    # データとラベルに切り分け(ラベルは不要)
    # 指定したnumber(ラベル)で抜き取り
    number = 5
    train_data, train_label = train._datasets
    #train_data = train_data[train_label == number]
    train_data = train_data

    test_data, test_label = test._datasets
    #test_data_n  = test_data[test_label == number]
    #test_label_n = test_label[test_label == number]
    #test_data  = np.concatenate((test_data_n[0:5],  test_data[0:5]))
    #test_label = np.concatenate((test_label_n[0:5], test_label[0:5]))
    test_data = test_data[:20]
    test_label = test_label[:20]

    # 学習の条件
    # エポックとミニバッチサイズ
    epoch = 200
    batchsize = 32
    # 隠れ層のユニット数
    hidden = [100,20,2]

    fd = './models/'
    # VAEの学習
    model = train_stacked(train_data, hidden, epoch, batchsize, fd, train_mode, act=chainer.functions.tanh)

    # 再構成を行う
    # AutoEncoderの再構成を行うクラスを定義
    ar = Reconst(model)
    feat_train, reconst_train, err_train = ar(train_data)
    feat_test,  reconst_test,  err_test  = ar(test_data)

    col = ['r','g','b','c','m','y','orange','black','gray','violet']
    c_list = []
    for i in range(train_label.shape[0]):
        c_list.append(col[train_label[i]])
        

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(3000):
        ax.scatter(x=feat_train[i,0],y=feat_train[i,1],marker='.',color=c_list[i])
    plt.show()


    rn = np.arange(-2,2,0.2)
    dt = []
    for i in range(20):
        for j in range(20):
            dt.append( np.array( [rn[i],rn[j]],np.float32) )
    dt=np.asarray(dt)

    rc = ar.decode(dt).data
    print(type(rc))
    
    fig = plt.figure()
    for i in range(0,400):
        #cv2.imshow('',rc[i].reshape((28,28)))
        ax1 = fig.add_subplot(20,20,i+1) 
        ax1.imshow(rc[i].reshape((28,28)),cmap='gray')
        #cv2.waitKey(1)
    plt.show()
    

