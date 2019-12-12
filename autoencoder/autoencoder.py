import os 
import sys
import numpy

# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torch.utils.data import DataLoader

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
            data = F.tanh(data)
        elif func == 'sigmoid':
            data = torch.sigmoid(data)
        elif func == 'relu':
            data = F.relu(data)
        return data

# **********************************************
# Training autoencoder by trainer
# **********************************************
def training_autoencoder(
        data, hidden, max_epoch, batchsize,
        fe='sigmoid', fd='sigmoid'):

    # log 
    log_interval = 50
    # GPUが使えればGPUを使う
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # input size
    inputs = data.shape[1]
    # number of data
    len_train = data.shape[0]

    # モデルの定義
    ae = Autoencoder(inputs, hidden, fe=fe, fd=fd)
    model = AutoencoderTrainer(ae, ae_method=ae_method, rho=rho, s=s)
    opt = optim.Adam(model.parameters())

    # 学習開始
    train(epochs=max_epoch, model=model,
          train_loader=train_loader, valid_loader=valid_loader,
          criterion=F.mse_loss, optimizer=opt,
          writer=log_writer, device=device, log_interval=log_interval)

    # モデル保存
    torch.save(model.state_dict(), './checkpoints/final_weights.pt')

    log_writer.close()

    # データの形式を変換する
    # train = datasets.TupleDataset(data, data)
    # train_iter = iterators.SerialIterator(train, batchsize)

    # training
    # updater = training.StandardUpdater(train_iter, opt, device=gpu_device)
    # trainer = training.Trainer(updater, (max_epoch, 'epoch'), out=out_dir)
    # trainer.extend(extensions.LogReport())
    # trainer.extend(extensions.PrintReport( ['epoch', 'main/loss']))
    # trainer.run()

    # GPU -> CPU
    if device is 'cuda':
        ae.to('cpu')

    return ae

# 指定した値で正規化
class Normalization(object):
    def __init__(self, value):
        self.value = value
    def __call__(self, data):
        size = data.shape
        if len(size) is 3:
            data = data.reshape(size[0], size[1]*size[2])
        return data/self.value

# **********************************************
# Sample: training MNIST by autoencoder
# **********************************************
def main():
 
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

    # 前処理の定義
    # trans = torchvision.transforms.Compose([
        # torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize((0.5,), (0.5,))])
    trans = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        Normalization(255)])

    # MNISTデータの読み込み
    #     root: MNISTデータが格納されているパスを設定
    #     train=Ture: 学習用のデータ取得
    #     download=True: MNISTデータがrootになければDL
    #     transform: 定義した前処理(設定すると自動で実行)
    train_dataset = torchvision.datasets.MNIST(
            root='../data', train=True, download=True, transform=trans)

    # 学習用にデータを分ける
    batchsize = 32
    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)

    model = Autoencoder(784, 100)
    opt = optim.Adam(model.parameters())

    max_epoch = 10
    for epoch in range(max_epoch):
        print(epoch + 1)
        for i, (d, l) in enumerate(train_loader, 0):
            d.to('cuda')
            y = model(d)
            loss = F.mse_loss(y, d)
            model.zero_grad()
            loss.backward()
            opt.step()

if __name__ == '__main__':
    main()


