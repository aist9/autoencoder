
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Chainer
import chainer
from chainer import serializers

# autoencoder.py
import variational_autoencoder
from variational_autoencoder import VariationalAutoEncoder, Reconst, training_vae
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
    
    train_data, train_label = train._datasets
    test_data, test_label = test._datasets

    test_data = test_data[:20]
    test_label = test_label[:20]

    # 学習の条件
    # エポックとミニバッチサイズ
    epoch = 200
    batchsize = 64
    # 隠れ層のユニット数
    hidden = [100,20,2]

    fd = './models/'
    # VAEの学習
    vae = training_vae(train_data, hidden, epoch, batchsize, \
             act_func='sigmoid', gpu_device=0, \
             loss_function='mse')
 

    # 再構成
    ar = Reconst(vae)
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


    rn = np.arange(-3,3,0.3)
    dt = []
    for i in range(20):
        for j in range(20):
            dt.append( np.array( [rn[i],rn[j]],np.float32) )
    dt=np.asarray(dt)

    rc = ar.decode(dt).data
    
    fig = plt.figure()
    for i in range(0,400):
        #cv2.imshow('',rc[i].reshape((28,28)))
        ax1 = fig.add_subplot(20,20,i+1) 
        ax1.imshow(rc[i].reshape((28,28)),cmap='gray')
        #cv2.waitKey(1)
    plt.show()
    

