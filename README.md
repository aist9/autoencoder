# autoencoder


## Overview

Autoencoder(AE)とVariational autoencoder(VAE)のサンプル

**現在はPyTorchへの移植作業も行っています**

## Requirement

- Essential
    - python3
    - numpy
    - matplotlib
    - sklearn
- Using chainer
    - chainer
    - cupy (training only)
    - opencv (VAE only)
- Using PyThorch
    - pytorch

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
        - torch 1.3.1 (using pytorch)

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
sys.path.append(os.path.join(home, 'workspace', 'autoencoder/autoencoder_by_chainer'))
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

