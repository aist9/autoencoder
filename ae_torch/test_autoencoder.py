"""
Unit test of autoencoder.py

Run
----
# using pytest (3rd-party)
pytest test_autoencoder.py
"""

import unittest
import torch.nn as nn
import torch
        
from autoencoder import AE
from autoencoder import Encoder_Decoder

import os

class TestAutoEncoder(unittest.TestCase):

    def setUp(self):

        # construction of Encoder_Decoder object
        hidden = [128, 2]
        act_func = 'tanh'
        out_func = 'sigmoid'
        use_BN = True
        init_method = nn.init.xavier_uniform_
        is_gauss_dist = 'True'
        device = 'cuda'
        self.ed = Encoder_Decoder(hidden, act_func, out_func, use_BN, init_method,
                             is_gauss_dist=is_gauss_dist, device=device)
  
        # construction of AutoEncoder object
        input_shape = 784
        fd = './model/torch'
        self.ae = AE(input_shape ,hidden, act_func=act_func, out_func=out_func,
                  use_BN=use_BN, folder=fd, is_gauss_dist=is_gauss_dist,
                  device=device)

        # torchのMNISTの呼び出し方がよくわからんかったのでchainerで代用
        # MNISTデータの読み込み
        import chainer
        self.train, self.test = chainer.datasets.get_mnist()
        # データとラベルに分割
        self.train_data, self.train_label = self.train._datasets
        self.test_data, self.selftest_label = self.test._datasets

            

    def test_Encoder_Decoder_init(self):
        # Check the construction of an Encoder_Decoder object
        hidden = [128, 2]
        act_func = 'tanh'
        out_func = 'sigmoid'
        use_BN = True
        init_method = nn.init.xavier_uniform_
        is_gauss_dist = 'True'
        device = 'cuda'
        ed = Encoder_Decoder(hidden, act_func, out_func, use_BN, init_method,
                             is_gauss_dist=is_gauss_dist, device=device)
        self.assertIsNotNone(ed)


    def test_AE_init(self):
        # Check the construction of a AE object
        input_shape = 784
        hidden = [128, 2]
        act_func = 'tanh'
        out_func = 'sigmoid'
        use_BN = True
        is_gauss_dist = 'True'
        device = 'cuda'
        fd = './model/torch'
        ae = AE(input_shape ,hidden, act_func=act_func, out_func=out_func,
                  use_BN=use_BN, folder=fd, is_gauss_dist=is_gauss_dist,
                  device=device)

        self.assertIsNotNone(ae)


    def test_train(self):
        train_mode = 'train'
        epoch = 3
        batchsize = 128
        before_state = self.ae.model.state_dict()
        self.ae.train(self.train_data[:10], epoch, batchsize, C=1.0, k=1,
                       valid=self.test_data, is_plot_weight=True)
        after_state = {key: value.to('cpu') for key, value in self.ae.model.state_dict().items()} 

        self.assertFalse(before_state.values() == after_state.values())

    """
    def test_retrain(self):
        train_mode = 'retrain'
        epoch = 3
        batchsize = 128

        self.ae.load_model()
        
        before_state = self.ae.model.state_dict()
        self.ae.train(self.train_data, epoch, batchsize, C=1.0, k=1, valid=None)
        after_state = {key: value.to('cpu') for key, value in self.ae.model.state_dict().items()} 

        self.assertFalse(before_state.values() == after_state.values())
    """
    
    def test_load_model(self):  
        self.assertTrue(callable(self.ae.load_model))
    
    def test_model_to_eval(self):    
        self.assertTrue(callable(self.ae.model_to_eval))
    
    
    def test_reconst(self):
        self.assertTrue(callable(self.ae.reconst))


    def test_featuremap_to_image(self):
        self.assertTrue(callable(self.ae.featuremap_to_image))
    



if __name__ == '__main__':
    unittest.main()
