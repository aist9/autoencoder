import argparse
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np

import chainer
from chainer import training
from chainer.training import extensions
import chainerx

import vae

# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy import io

from os.path import expanduser
home = expanduser("~")

if __name__ == '__main__':

    # コマンドライン引数を読み込み
    # 引数が'-1'なら学習しない
    args = sys.argv
    train_mode = True 
    if 2 <= len(args):
        if args[1] == '-1':
            train_mode = False
 
    # データの読み込み
    data_file_name = 'FFT16384_FFTdata.mat'
    reject_file_name = 'rejectnum.csv'
    data_file_path = os.path.join(home, 'workspace/dataset/pump', data_file_name)
    reject_file_path = os.path.join(home, 'workspace/dataset/pump', reject_file_name)
    matdata = io.loadmat(data_file_path, squeeze_me=True)
    mat_data = matdata["data"]

    data = []
    reject_number = (np.loadtxt(reject_file_path) - 1).tolist()
    for i, d in enumerate(mat_data):
        if reject_number.count(i) == 0:
            data.append(d)
    data = data[0:-1] 
    data = np.array(data, dtype='float32')
    print(data.shape)

    data_max = np.max(np.abs(data))
    data = data / data_max + 1

    # trainとtestに分ける
    train = data[0:2000]
    test = data[2000:]
    
    model = vae.training_vae(train, data.shape[1], 100, 1, beta=1.0, k=1, device=0,
                epoch=500, batch=100)

    chainer.serializers.save_npz('model.npy', model)

    feat, reconst, err, mu, ln_sigma, z = vae.reconst(model, data)

    print(feat.shape)
    print(reconst.shape)
    print(err.shape)
    print(mu.shape)
    print(ln_sigma.shape)
    print(z.shape)

    # plt.plot(err)
    # plt.show()

    # plt.plot(data[0, :])
    # plt.plot(reconst[0, :])
    # plt.show()

    plt.plot(mu)
    plt.show()

    print(np.max(train))
    print(np.min(train))
    print(np.max(reconst))
    print(np.min(reconst))

