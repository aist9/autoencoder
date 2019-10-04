
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.computational_graph as c
from chainer import Variable, optimizers, initializer,cuda
from chainer.functions.loss.vae import gaussian_kl_divergence

import cupy

import os

import numpy as np

import chainer
import chainer.distributions as D
import chainer.functions as F
import chainer.links as L
from chainer import reporter
from chainer import training
from chainer.training import extensions
import chainerx

class AvgELBOLoss(chainer.Chain):
    """Loss function of VAE.
    The loss value is equal to ELBO (Evidence Lower Bound)
    multiplied by -1.
    Args:
        encoder (chainer.Chain): A neural network which outputs variational
            posterior distribution q(z|x) of a latent variable z given
            an observed variable x.
        decoder (chainer.Chain): A neural network which outputs conditional
            distribution p(x|z) of the observed variable x given
            the latent variable z.
        prior (chainer.Chain): A prior distribution over the latent variable z.
        beta (float): Usually this is 1.0. Can be changed to control the
            second term of ELBO bound, which works as regularization.
        k (int): Number of Monte Carlo samples used in encoded vector.
    """

    def __init__(self, encoder, decoder, prior, beta=1.0, k=1):
        super(AvgELBOLoss, self).__init__()
        self.beta = beta
        self.k = k

        with self.init_scope():
            self.encoder = encoder
            self.decoder = decoder
            self.prior = prior

    def __call__(self, x):
        q_z = self.encoder(x)
        z = q_z.sample(self.k)
        p_x = self.decoder(z)
        p_z = self.prior()

        self.q_z = q_z
        self.z = z
        self.p_x = p_x
        self.p_z = p_z

        # reconstr = F.mean(p_x.log_prob(
            # F.broadcast_to(x[None, :], (self.k,) + x.shape)))
        reconstr = F.mean_squared_error(x, p_x[0,:,:])
        kl_penalty = F.mean(chainer.kl_divergence(q_z, p_z))
        loss = reconstr + self.beta * kl_penalty
        reporter.report({'loss': loss}, self)
        reporter.report({'reconstr': reconstr}, self)
        reporter.report({'kl_penalty': kl_penalty}, self)
        return loss

class Encoder(chainer.Chain):

    def __init__(self, n_in, n_latent, n_h):
        super(Encoder, self).__init__()
        with self.init_scope():
            self.linear = L.Linear(n_in, n_h)
            self.mu = L.Linear(n_h, n_latent)
            self.ln_sigma = L.Linear(n_h, n_latent)

    def forward(self, x, get_variable=False):
        # h = F.tanh(self.linear(x))
        h = F.sigmoid(self.linear(x))
        mu = self.mu(h)
        ln_sigma = self.ln_sigma(h)  # log(sigma)
        if get_variable:
            return h, mu, ln_sigma, D.Independent(D.Normal(loc=mu, log_scale=ln_sigma))
        else:
            return D.Independent(D.Normal(loc=mu, log_scale=ln_sigma))


class Decoder(chainer.Chain):

    def __init__(self, n_in, n_latent, n_h, binary_check=False):
        super(Decoder, self).__init__()
        self.binary_check = binary_check
        with self.init_scope():
            self.linear = L.Linear(n_latent, n_h)
            self.output = L.Linear(n_h, n_in)

    def forward(self, z, inference=False):
        n_batch_axes = 1 if inference else 2
        h = F.sigmoid(self.linear(z, n_batch_axes=n_batch_axes))
        # h = F.tanh(self.linear(z, n_batch_axes=n_batch_axes))
        h = self.output(h, n_batch_axes=n_batch_axes)
        return h
        # if get_variable or :
            # return h
        # else:
            # return D.Independent(
                # D.Bernoulli(logit=h, binary_check=self.binary_check),
                # reinterpreted_batch_ndims=1)

class Prior(chainer.Link):

    def __init__(self, n_latent):
        super(Prior, self).__init__()

        dtype = chainer.get_dtype()
        self.loc = np.zeros(n_latent, dtype)
        self.scale = np.ones(n_latent, dtype)
        self.register_persistent('loc')
        self.register_persistent('scale')

    def forward(self):
        return D.Independent(
            D.Normal(self.loc, scale=self.scale), reinterpreted_batch_ndims=1)


def make_encoder(n_in, n_latent, n_h):
    return Encoder(n_in, n_latent, n_h)


def make_decoder(n_in, n_latent, n_h, binary_check=False):
    return Decoder(n_in, n_latent, n_h, binary_check=binary_check)


def make_prior(n_latent):
    return Prior(n_latent)

# VAEの学習
def training_vae(train, inputs, hidden, z, beta=1.0, k=1, device=0,
                epoch=100, batch=100):

    # setting network
    encoder = make_encoder(inputs, z, hidden)
    # decoder = make_decoder(inputs, z, hidden)
    decoder = make_decoder(inputs, z, hidden)
    prior = make_prior(z)
    avg_elbo_loss = AvgELBOLoss(encoder, decoder, prior,
                                    beta=beta, k=k)
    avg_elbo_loss.to_device(device)

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(avg_elbo_loss)

    # 
    train_iter = chainer.iterators.SerialIterator(train, batch)
    # test_iter = chainer.iterators.SerialIterator(test, args.batch_size,
                                                 # repeat=False, shuffle=False)

    # Set up an updater. StandardUpdater can explicitly specify a loss function
    # used in the training with 'loss_func' option
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=device, loss_func=avg_elbo_loss)

    # Set up the trainer and extensions.
    trainer = training.Trainer(updater, (epoch, 'epoch'), out='results')

    trainer.extend(extensions.snapshot(), trigger=(epoch, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 
         'main/reconstr', 'main/kl_penalty', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar())

    # Run the training
    trainer.run()

    return avg_elbo_loss.to_cpu()

# 再構成
def reconst(model, data):
    
    # 特徴量と再構成データの取得
    feat, mu, ln_sigma, z = model.encoder.forward(data, get_variable=True)
    reconst = model.decoder.forward(z.sample(1))
    reconst = reconst[0, :, :]

    # 再構成誤差の算出
    err = np.sum((data - reconst.data) ** 2, axis=1) / data.shape[1]

    return feat.data, reconst.data, err, mu.data, ln_sigma.data, (z.sample(1)).data

