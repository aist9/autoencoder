# autoencoder

## Overview

Autoencoder(AE)とVariational autoencoder(VAE)のサンプル

**現在はPyTorchへの移植作業も行なっています**

## Requirement

- python3
- numpy
- matplotlib
- chainer
- cupy (training only)
- opencv (VAE only)

動作確認済環境
- Mac
    - macOS Catalina 10.15.1
    - Cahiner 5.0.0
    - NumPy 1.17.0
- Linux
    - Ubuntu 18.04.3 LTS
    - Cahiner 6.2.0
    - Cupy-cuda101 6.2.0
    - NumPy 1.17.0

学習時はGPU環境がおすすめです

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

## autoencoder_by_chainer

AEとVAEのファイルがあります

Chainerで実装されています

- autoencoder.py
- variational_autoencoder.py

各ファイルの中にはいくつかのクラス、関数、サンプルが書かれたmain文があります


