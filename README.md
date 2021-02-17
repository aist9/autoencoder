# autoencoder


## Overview

Autoencoder(AE)とVariational autoencoder(VAE)のサンプル

**現在はPyTorchへの移植作業を行っています**

Chainer版(ae_chain)は基本的にメンテナンスのみ、PyTorch版(ae_torch)が開発中です

## Requirement

- Essential
    - Common
        - python3
        - numpy
        - matplotlib
    - Using chainer
        - chainer
        - cupy (training only)
        - opencv (VAE only)
    - Using PyThorch
        - pytorch
        - ignite
        - torchvision

- Certified environment
    - Mac
        - macOS Catalina 10.15.1
        - python 3.7.1
        - numpy 1.17.0
        - chainer 5.0.0 (using chainer)
    - Linux
        - Ubuntu 18.04.3 LTS
        - python 3.7.2
        - numpy 1.17.0
        - chainer 6.2.0 (using chainer)
        - cupy-cuda101 6.2.0 (using chainer)
        - pytorch 1.3.1 (using pytorch)
        - pytorch-ignite 0.2.1
        - torchvision 0.4.2

## Installation
```
$ git clone https://github.com/aist9/autoencoder
```

使いたいファイル内でimport(以下はimportの例です)

```
import autoencoder
from autoencoder import Reconst, Autoencoder, training_autoencoder
```
違う階層に置く場合は適宜パスを通す
```
# "~/workspace"に存在する場合
from os.path import expanduser
home = expanduser("~")
sys.path.append(os.path.join(home, 'workspace', 'autoencoder/ae_chain'))
```

## ae_chain

Chainerで実装されたAEとVAEのファイルがあります

- autoencoder.py
- variational_autoencoder.py

各ファイルの中にはいくつかのクラス、関数、サンプルが書かれたmain文があります

## ae_torch

PyTorchで実装されたAEのファイルがあります

現在 ae_chain/autoencoder.py から移植している段階です

- autoencoder.py

各ファイルの中にはいくつかのクラス、関数、サンプルが書かれたmain文があります

## vae_chain

Chainerで実装されたVAEのファイルがあります

- variational_autoencoder.py
- variational_autoencoder_ed.py
- vae_gaussian.py
- vae_gaussian_ed.py

ae_chainのものと実装を変更しています
"_ed"が付いているものは, encoderとdecoderが独立しており, optimizerが個々に設定できます
"_gaussian"が付いているものは, decoderの出力が2出力になっており, loss関数もbernoulli_nllからgaussian_nllに変更しています
詳しくはvae_gaussian_ed.pyに書いてあります

## vae_torch

PyTorchで実装されたVAEのファイルがあります

- variational_autoencoder.py
- variational_autoencoder_ed.py
- sample.py

こちらもvae_chainのものと同様ですが, "_gaussian"の切り替えをそれぞれのプログラム中で行えるようにしました
また, およそvae_chainの実装に習っていますが, こちらは"_ed"版は無印版を可能な限り継承するように実装しました
サンプルや使用方法は"sample.py"を参考にしてください

