# autoencoder

## Overview

Autoencoder(AE)とVariational autoencoder(VAE)のサンプル

## Requirement

- python3
- numpy
- matplotlib
- chainer
- cupy
- opencv (VAE only)

## Installation

Githubからclone
```
$ git clone https://github.com/aist9/autoencoder
```

使いたいファイル内でimport
```
import autoencoder
from autoencoder import Reconst, Autoencoder, training_autoencoder, train_stacked
```

違う階層に置く場合は適宜パスを通す

```
# "~/workspace"に存在する場合
from os.path import expanduser
home = expanduser("~")
sys.path.append(os.path.join(home, 'workspace', 'autoencoder/scripts'))
```

## scripts

AEとVAEのファイルがあります

- autoencoder.py
- variational_autoencoder.py

各ファイルの中にはいくつかのクラス、関数、サンプルが書かれたmain文があります

