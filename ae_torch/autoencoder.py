import os 
import numpy as np
import matplotlib.pyplot as plt

import json

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
# AutoEncoder model
# **********************************************
class Encoder_Decoder(nn.Module):

    def __init__(self, layers, act_func=torch.tanh, out_func=torch.sigmoid,
                use_BN=False, init_method=nn.init.xavier_uniform_,
                is_gauss_dist=False ,device='cuda'):

        super().__init__()

        self.use_BN = use_BN
        self.act_func = act_func
        self.out_func = out_func
        self.is_gauss_dist = is_gauss_dist
        self.device = device
        self._makeLayers(layers, init_method)

    def _makeLayers(self, hidden, init_method):

        encode_layer = []
        decode_layer = []

        e = nn.Linear(hidden[0], hidden[1])
        d = nn.Linear(hidden[1], hidden[0])        

        init_method(e.weight)
        init_method(d.weight)

        encode_layer.append(e)
        decode_layer.append(d)

        if self.use_BN:
            e = nn.BatchNorm1d(hidden[1])
            d = nn.BatchNorm1d(hidden[0])

            encode_layer.append(e)
            decode_layer.append(d)
        
        
        self.encode_layer = nn.ModuleList(encode_layer)
        self.decode_layer = nn.ModuleList(decode_layer)
        
    def __call__(self, x): 
        h = self.encode(x)
        d_out = self.decode(h)
        return h, d_out


    def encode(self, x):
        e = x
        for i in range(len(self.encode_layer)):
            e = self.encode_layer[i](e) if self.use_BN and not (i & 1) \
                else self.act_func(self.encode_layer[i](e))
        return e

    def decode(self, h):
        d_out = h
        for i in range(len(self.decode_layer)):
            d_out = self.decode_layer[i](d_out) if self.use_BN and not (i & 1) \
                else self.act_func(self.decode_layer[i](d_out))
        return d_out
        

    def reconst_loss(self, x, dec_out):
        reconst = F.mse_loss(x, dec_out)
        return reconst


# **********************************************
# Autoencoder class
# **********************************************
class AE():
    #def __init__(self, inputs, hidden, fe='sigmoid', fd='sigmoid'):
    def __init__(self, input_shape, hidden, act_func=torch.tanh,
                 out_func=torch.sigmoid, use_BN=False, init_method='xavier',
                 folder='./model', is_gauss_dist=False, device='cuda'):

        activations = {
                "sigmoid"   : torch.sigmoid, \
                "tanh"      : torch.tanh,    \
                "softplus"  : F.softplus,    \
                "relu"      : torch.relu,    \
                "leaky"     : F.leaky_relu,  \
                "elu"       : F.elu,         \
                "identity"  : lambda x:x     \
        }
        
        self.device = device

        # Specify the activation function
        if isinstance(act_func, str):
            if act_func in activations.keys():
                act_func = activations[act_func]
            else:
                print('arg act_func is ', act_func, '. This value is not exist. \
                      This model uses identity function as activation function.')
                act_func = lambda x: x
        
        if isinstance(out_func, str):
            if out_func in activations.keys():
                out_func = activations[out_func]
            else:
                print('arg out_func is ', out_func, '. This value is not exist. \
                      This model uses identity function as activation function.')
                out_func = lambda x: x
        
        if out_func != torch.sigmoid:
            print('※ out_func should be sigmoid.')


        # Specify the initialization method
        if isinstance(init_method, str):
            inits = {
                    "xavier"    : nn.init.xavier_uniform_, \
                    "henormal"  : nn.init.kaiming_uniform_ \
            }
            if init_method in inits.keys():
                init_method = inits[init_method]
            else:
                init_method = nn.init.xavier_uniform_
                print('init_method is xavier initializer')


        if not isinstance(hidden, list):
            hidden = [hidden]
        hidden = [input_shape] + hidden

        print('layer' + str(hidden))

        # A path for saving a trained model
        self.save_dir  = os.path.join(folder, '{}'.format(hidden))
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.save_dir+'/weight_plot', exist_ok=True)

        self.is_gauss_dist = is_gauss_dist
        self.set_model(hidden, act_func, out_func, use_BN, init_method, is_gauss_dist=is_gauss_dist, device=self.device)
        self.set_optimizer()

        # Number of weights used in weight_plot function
        self.weight_num = len(hidden) + 1

    def __call__(self, x):
        return self.model(x)
        
     # train
    def train(self,train,epoch,batch,C=1.0, k=1, valid=None, is_plot_weight=False):

        if valid is None:
            print('epoch\tloss\t\treconst\t\tMSE')
        else:
            print('epoch\tloss\t\treconst\t\tMSE\t\tvalid_MSE')


        # conversion data
        train_data = torch.Tensor(train)
        dataset = torch.utils.data.TensorDataset(train_data, train_data)
        train_loader = DataLoader(dataset, batch_size=batch, shuffle=True)

        # trainer
        trainer = self.trainer(C=C,k=k, device=self.device)

        # log variables init.
        log = []
        rec_loss_iter = []

        # executed function per iter
        @trainer.on(Events.ITERATION_COMPLETED)
        def add_loss(engine):
            rec_loss_iter.append(engine.state.output)                        
        
        # executed function per epoch
        @trainer.on(Events.EPOCH_COMPLETED)
        def log_report(engine):
            epoch = engine.state.epoch            
            rec_loss = np.mean(rec_loss_iter)
            log.append({'epoch':epoch, 'reconst':rec_loss})
            if epoch % 10 == 0 or epoch==1:
                perm = np.random.permutation(len(train))[:batch]
                mse = self.MSE(train[perm]).mean()
                if valid is None:
                    print(f'{epoch}\t{rec_loss:.6f}\t{mse:.6f}')
                else:
                    val_mse = self.MSE(valid).mean()
                    print(f'{epoch}\t{rec_loss:.6f}\t{mse:.6f}\t{val_mse:.6f}')

                if is_plot_weight: # output layer weight.
                    self.plot_weight(epoch)
            
            rec_loss_iter.clear()            

        # start training
        trainer.run(train_loader, max_epochs=epoch)

        # save model weight
        self.save_model()
        
        # log output
        file_path = os.path.join(self.save_dir, 'log')
        file_ = open(file_path, 'w')
        json.dump(log, file_, indent=4)
    

    def trainer(self, C=1.0, k=1, device=None):

        self.model_to(device)

        def prepare_batch(batch, device=None):
            x, y = batch
            return (convert_tensor(x, device=device),
                    convert_tensor(y, device=device))

        def _update(engine, batch):
            self.zero_grad()
            x, y = prepare_batch(batch, device=device)
            h = self.encode(x)                            
            
            reconst_loss = 0

            for l in range(k):                
                d_out = self.decode(h)
                reconst_loss += self.reconst_loss(y, d_out) / float(k)
            loss = reconst_loss
            loss.backward()
            self.grad_clip()
            self.step()
            return loss.item()
 
        return Engine(_update)   


    # For overriding to omit writing code for inheritance
    def encode(self, x):
        return self.model.encode(x)
    
    def decode(self, z):
        return self.model.decode(z)
    
    def reconst_loss(self, x, d_out):
        return self.model.reconst_loss(x, d_out)
    
    def set_model(self, hidden, act_func, out_func, use_BN, init_method,
                 is_gauss_dist, device):
        self.model = Encoder_Decoder(hidden, act_func, out_func, use_BN,
                     init_method, is_gauss_dist=is_gauss_dist, device=device)
    
    def set_optimizer(self, learning_rate=0.001, beta1=0.9, beta2=0.999,
                      weight_decay=0, gradient_clipping=None):
        betas=(beta1, beta2)
        self.opt = optim.Adam(self.model.parameters(), lr=learning_rate,
                              betas=betas, eps=1e-08, weight_decay=weight_decay,
                              amsgrad=False)        
        self.gradient_clipping = gradient_clipping

    def model_to(self, device):
        self.model.to(device)
    
    def zero_grad(self):
        self.opt.zero_grad()
    
    def step(self):
        self.opt.step()

    def grad_clip(self):
        if self.gradient_clipping is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           self.gradient_clipping)


    # 各レイヤーの重みをプロット. 重み更新が機能してるか確認.
    def plot_weight(self, epoch):
        dic = self.model.state_dict()

        fig = plt.figure(figsize=(16,8))
        plot_num = 0
        for k in dic.keys():
            if 'weight' in k:
                plot_num += 1
                plot_data = self.tensor_to_np(dic[k]).reshape(-1)
                plt.subplot(2,self.weight_num,plot_num)
                plt.plot(plot_data, label=k)
                plt.legend()
        plt.tight_layout()
        
        plt.close()

    # modelの保存. trainメソッドの最後に呼び出される.
    def save_model(self, path=None):
        path = self.save_dir if path is None else path
        torch.save(self.model.state_dict(), path+'/model.pth')

    # modelのロード. 学習済みモデルを使用したい場合に呼び出す.
    def load_model(self, path=None):
        path = self.save_dir if path is None else path
        param = torch.load( path + '/model.pth')
        self.model.load_state_dict(param)
        self.model.to(self.device)

    # evalモードに切り替え
    def model_to_eval(self):
        self.model.eval()

    def reconst(self, data, unregular=False):
    # かきかえ
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if not isinstance(data, torch.Tensor):
            data = self.np_to_tensor(data)

        h = self.encode(data)
        d_out = self.decode(h)

        rec = d_out[0] 
        mse = torch.mean((rec - data) ** 2, dim=1)

        h = self.tensor_to_np(h)
        rec = self.tensor_to_np(rec)
        mse = self.tensor_to_np(mse)

        return h, rec, mse

        # Calcurate Mean square error

    def MSE(self, data):
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if not isinstance(data, torch.Tensor):
            data = self.np_to_tensor(data)

        e, d_out = self(data)
        rec = d_out[0] if self.is_gauss_dist else d_out
        mse = torch.mean((rec - data) ** 2, dim=1)
        return self.tensor_to_np(mse)

    def featuremap_to_image(self, feat):
        if feat.ndim == 1:
            feat = feat.reshape(1, -1)
        if not isinstance(feat, torch.Tensor):
            feat = self.np_to_tensor(feat)

        d_out = self.decode(feat)
        d_out = d_out[0] if self.is_gauss_dist else d_out
        return self.tensor_to_np(d_out)

    # ndarray -> torch.Tensor
    def np_to_tensor(self, data):
        return torch.Tensor(data).to(self.device)

    # torch.Tensor -> ndarray
    def tensor_to_np(self, data):
        return data.detach().to('cpu').numpy()

