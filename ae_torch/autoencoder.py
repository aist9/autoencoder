import os 
import sys
import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

# pytorch
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.contrib.handlers.tensorboard_logger import *

# **********************************************
# autoencoder class
# **********************************************
class Autoencoder(nn.Module):
    def __init__(self, inputs, hidden, fe='sigmoid', fd='sigmoid'):
        super(Autoencoder, self).__init__()
        self.le = nn.Linear(inputs, hidden)
        self.ld = nn.Linear(hidden, inputs)

        # 活性化関数の指定
        self.fe = fe
        self.fd = fd

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
        if self.fe is not None:
            h = self.activation_function(h, self.fe)
        return h
        
    # Decoder
    def decoder(self, h):
        y = self.ld(h)
        if self.fd is not None:
            y = self.activation_function(y, self.fd) 
        return y
    
    # 活性化関数
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
def loss_function(data, target):
    return F.mse_loss(data, target)

class LossFunction(object):
    def __init__(self):
        pass
    def __call__(self, data, target):
        return F.mse_loss(data, target)

# **********************************************
# Training autoencoder by trainer
# **********************************************
def training_autoencoder_(
        data, hidden, max_epoch, batchsize,
        fe='sigmoid', fd='sigmoid'):

    # gpu setting
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    
    # input size
    inputs = data.shape[1]

    # numpy -> tensor -> DataLoader
    train_data = torch.Tensor(data)
    train_data.to(device)
    dataset = torch.utils.data.TensorDataset(train_data, train_data)
    train_loader = DataLoader(dataset, batch_size=batchsize, shuffle=True)

    # Define the autoencoder model
    model = Autoencoder(inputs, hidden)
    opt = optim.Adam(model.parameters())

    # loss function
    criterion = LossFunction()

    # trainer
    trainer = create_supervised_trainer(
            model, opt, criterion, device=device)

    # print loss value (each epoch)
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_report(engine):
        if engine.state.epoch == 1:
            print('epoch\t\tloss')
        print(f"{engine.state.epoch}\t\t{engine.state.output:.10f}")

    # start training
    trainer.run(train_loader, max_epochs=max_epoch)
    
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
    save_dir = '../output/result_autoencoder'
    os.makedirs(save_dir, exist_ok=True)

    # モデルの保存パス
    model_path = os.path.join(save_dir, 'autoencoder_torch')

    # -------------------------------------
    # データの準備
    # -------------------------------------

    # 指定した数字データを抜くための変数
    number = 0

    # MNISTのデータセットを加工がしやすいようにsklearnで取得
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

    # モデルの定義
    
    # コマンドライン引数が'-1'の場合学習しない
    if train_mode is True:
        # Autoencoderの学習
        model = training_autoencoder_(train_data, hidden, epoch, batchsize)
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
    reconst_test = y.detach().clone().numpy()

    # 保存先ディレクトリの生成
    save_dir = os.path.join(save_dir, 'img_torch')
    os.makedirs(save_dir, exist_ok=True)

    # -------------------------------------
    # 可視化
    # -------------------------------------
    
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

