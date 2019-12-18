import os 
import sys
import numpy as np
import matplotlib.pyplot as plt

import json
import math

# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from ignite.engine import Engine, Events
from ignite.utils import convert_tensor
from ignite.contrib.handlers.tensorboard_logger import *

# **********************************************
# autoencoder class
# **********************************************
class Autoencoder(nn.Module):
    def __init__(self, inputs, hidden, fe='sigmoid', fd='sigmoid'):
        super(Autoencoder, self).__init__()
        self.le = nn.Linear(inputs, hidden)
        self.ld = nn.Linear(hidden, inputs)

        self.fe = fe
        self.fd = fd

    # Forward
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
        if self.fe is not None:
            h = self.activation_function(h, self.fe)
        return h
        
    # Decoder
    def decoder(self, h):
        y = self.ld(h)
        if self.fd is not None:
            y = self.activation_function(y, self.fd) 
        return y
    
    # Activation function
    def activation_function(self, data, func): 
        if func == 'tanh':
            data = torch.tanh(data)
        elif func == 'sigmoid':
            data = torch.sigmoid(data)
        elif func == 'relu':
            data = torch.relu(data)
        return data

# **********************************************
# Loss function
# **********************************************
class LossFunction(object):
    def __init__(self, ae_method=None, rho=0.05, s=0.001):
        # calculation setting
        self.ae_method = ae_method
        self.rho = rho 
        self.s = s

    # calculate loss
    def __call__(self, y, h, t):
        # mse loss
        loss = F.mse_loss(y, t)

        # For sparse AE
        if self.ae_method == 'sparse':
            if 0 < self.s:
                kld = self.reg_sparse(h)
                loss += self.s * kld

        return loss

    # sparse regularization term
    def reg_sparse(self, h):
        rho_hat = torch.sum(h, axis=0) / h.shape[0]
        kld = torch.sum(self.rho * torch.log(self.rho / rho_hat) + \
                    (1 - self.rho) * torch.log((1 - self.rho) / (1 - rho_hat)))
        return kld

# **********************************************
# Trainer Wrraper
# **********************************************
def nn_trainer(
        model, optimizer, loss_function, device=None):

    if device:
        model.to(device)

    def prepare_batch(batch, device=None):
        x, y = batch
        return (convert_tensor(x, device=device),
                convert_tensor(y, device=device))

    def _update(engine, batch):
        optimizer.zero_grad()
        x, t = prepare_batch(batch, device=device)
        y, h = model(x, hidden_out=True)
        loss = loss_function(y, h, t)
        loss.backward()
        optimizer.step()
        return loss.item()

    return Engine(_update)

# **********************************************
# Training autoencoder by trainer
# **********************************************
def training_autoencoder(
        data, hidden, max_epoch, batchsize,
        fe='sigmoid', fd='sigmoid',
        ae_method=None, rho=0.05, s=0.001,
        out_dir='result'):

    # gpu setting
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    
    # input size
    inputs = data.shape[1]
    
    # conversion data
    train_data = torch.Tensor(data)
    dataset = torch.utils.data.TensorDataset(train_data, train_data)
    train_loader = DataLoader(dataset, batch_size=batchsize, shuffle=True)

    # Define the autoencoder model
    model = Autoencoder(inputs, hidden)
    opt = optim.Adam(model.parameters())

    # loss function
    loss_function = LossFunction(ae_method=ae_method, rho=rho, s=s)

    # trainer
    trainer = nn_trainer(model, opt, loss_function, device=device)

    # log variables init.
    log = []
    loss_iter = []

    # add loss (each iter.)
    @trainer.on(Events.ITERATION_COMPLETED)
    def add_loss(engine):
        loss_iter.append(engine.state.output)
        
    # print loss value (each epoch)
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_report(engine):
        epoch = engine.state.epoch
        loss = sum(loss_iter) / len(loss_iter)
        log.append({'epoch':epoch,'loss':loss})
        if engine.state.epoch == 1:
            print('epoch\t\tloss')
        print(f'{epoch}\t\t{loss:.10f}')
        loss_iter.clear()

    # start training
    trainer.run(train_loader, max_epochs=max_epoch)
    
    # log output
    file_path = os.path.join(out_dir, 'log')
    file_ = open(file_path, 'w')
    json.dump(log, file_, indent=4)

    # gpu -> cpu
    if device is not 'cpu':
        model.to('cpu')

    return model

# **********************************************
# Viasualization
# **********************************************
# 重みの可視化
def weight_plot(
        model, plot_num, select_layer='encoder',
        reshape_size=None, save_path='w.png'):

    # パラメータの抽出
    param = model.state_dict()

    # 重みの抽出
    if select_layer is 'encoder':
        weight = param['le.weight'].clone().numpy()
    else:
        # decoderの場合は転置が必要
        weight = param['ld.weight'].clone().numpy().T

    # 自動でsubplotの分割数を決める
    row = int(np.sqrt(plot_num))
    mod = plot_num % row
    
    # 保存形式が.pdfの場合の処理
    if save_path in '.pdf':
        pp = PdfPages(save_path)

    # plot
    for i in range(plot_num):
        # 次の層i番目に向かう重みの抜き出し
        w = weight[i]
        # reshape(指定があれば)
        if reshape_size is not None:
            w = w.reshape(reshape_size)
        # 自動でsubplotの番号を与えplot
        plt.subplot(row, row+mod, 1+i)
        plt.imshow(w, cmap='gray')
    
    # 保存処理
    if save_path in '.pdf':
        pp.savefig()
        plt.close()
        pp.close()
    else:
        plt.savefig(save_path)
        plt.close()

# バイアスの可視化
def bias_plot(model, select_layer='encoder',
              reshape_size=None, save_path='w.png'):

    # パラメータの抽出
    param = model.state_dict()

    # バイアスの抽出
    if select_layer is 'encoder':
        bias = param['le.bias'].clone().numpy()
    else:
        bias = param['ld.bias'].clone().numpy().T

    # reshape(指定があれば)
    if reshape_size is not None:
        bias = bias.reshape(reshape_size)

    # 保存形式が.pdfの場合の処理
    if save_path in '.pdf':
        pp = PdfPages(save_path)
    # plot
    plt.imshow(bias, cmap='gray')

    # 保存処理
    if save_path in '.pdf':
        pp.savefig()
        plt.close()
        pp.close()
    else:
        plt.savefig(save_path)
        plt.close()

# **********************************************
# Sample: training MNIST by autoencoder
# **********************************************
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
    save_dir = '../output/result_ae_torch'
    os.makedirs(save_dir, exist_ok=True)

    # モデルの保存パス
    model_path = os.path.join(save_dir, 'autoencoder_torch')

    # -------------------------------------
    # データの準備
    # -------------------------------------

    # 指定した数字データを抜くための変数
    number = 0

    # MNISTのデータセットを加工がしやすいようにsklearnで取得
    from sklearn.datasets import fetch_openml
    mnist_data = fetch_openml('mnist_784', version=1, data_home='../data')
 
    # 前処理: 0 - 1の範囲になるように正規化
    data = mnist_data.data / 255
    label = mnist_data.target
    
    # 学習とテストに分割
    train_data = data[0:60000, :]
    train_label = np.asarray(label[0:60000], dtype='int32')
    test_data = data[60000:, :]
    test_label = np.asarray(label[60000:], dtype='int32')

    # 学習データ: 特定の番号のみ抽出したデータを用いる
    train_data = train_data[train_label==number]
    train_label = train_label[train_label==number]

    # -------------------------------------
    # 学習の準備
    # -------------------------------------

    # エポック
    epoch = 100
    # ミニバッチサイズ
    batchsize = 50
    # 隠れ層のユニット数
    hidden = 10

    # -------------------------------------
    # AutoEncoderの学習
    # -------------------------------------

    # コマンドライン引数が'-1'の場合学習しない
    if train_mode is True:
        # Autoencoderの学習
        model = training_autoencoder(
                train_data, hidden, epoch, batchsize,
                out_dir=save_dir)
        # モデルの保存
        torch.save(model.state_dict(), model_path)
    else:
        # 保存したモデルから読み込み
        model = Autoencoder(784, hidden)
        param = torch.load(model_path)
        model.load_state_dict(param)

    # -------------------------------------
    # 再構成
    # -------------------------------------

    y = model(torch.Tensor(test_data))
    y.to('cpu')

    h = model.encoder(torch.Tensor(test_data))
    h.to('cpu')

    reconst_test = y.detach().clone().numpy()

    feat_test = h.detach().clone().numpy()

    print(feat_test[0])
    

    # -------------------------------------
    # 可視化
    # -------------------------------------
    
    # 保存先ディレクトリの生成
    save_dir = os.path.join(save_dir, 'img_torch')
    os.makedirs(save_dir, exist_ok=True)

    # 1次元に並んだデータをreshapeするサイズ
    reshape_size = [28, 28]
    for (save_name, dataset) in zip(
            ['input', 'reconst'], [test_data, reconst_test]):
        # 入出力をplot
        for i, d in enumerate(dataset):
            save_path = os.path.join(
                    save_dir,
                    '_'.join([save_name, str(i+1)]) + '.png')
            d = d.reshape(reshape_size)
            plt.imshow(d, cmap='gray')
            plt.savefig(save_path)
            if i+1 == 10:
                break

    # # encoderの重みを可視化
    save_path = os.path.join(save_dir, 'encoder_weight.png')
    weight_plot(
            model, hidden, select_layer='encoder',
            reshape_size=[28, 28], save_path=save_path)
    
    # decoderの重みを可視化
    save_path = os.path.join(save_dir, 'decoder_weight.png')
    weight_plot(
            model, hidden, select_layer='decoder',
            reshape_size=[28, 28], save_path=save_path)
    
    # decoedrのバイアスを可視化
    save_path = os.path.join(save_dir, 'decoder_bias.png')
    bias_plot(model, select_layer='decoder',
              reshape_size=[28, 28], save_path=save_path)

if __name__ == '__main__':
    main()

