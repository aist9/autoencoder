# autoencoder

## Overview
python_modules(private)内に存在していたautoencoderを切り離し整備しているところ

## Requirement

- python3
- numpy
- matplotlib
- chainer
- cupy

## Installation

Githubからcloneする
```
$ git clone https://github.com/aist9/autoencoder
```

使いたいファイル内でimportする
```
import autoencoder
from autoencoder import Reconst, Autoencoder, training_autoencoder, train_stacked
```

違う階層に置く場合は適宜パスを通して使う


## scripts

オートエンコーダのモジュールと、サンプルファイルが入っています

- autoencoder.py
- sample.py
- variational_autoencoder.py
- vae_sample.py

autoencoder.pyの中にはいくつかのクラスがあります

## output

sample.pyを実行すると学習したモデルファイルが生成されます

## scripts/autoencoder.py

オートエンコーダの定義、学習済みモデルを使った再構成、学習をするための関数が用意

引数を与えることでスパース化にも対応している


階層数とリストの要素数を合わせる必要

## 
### class Autoencoder

```
# inputs: 入力サイズ
# hidden: 隠れ層サイズ
# fe:     Encoder(入力層→隠れ層)の活性化関数(Default is 'sigmoid')
# fd:     Decoder(隠れ層→出力層)の活性化関数(Default is 'sigmoid')
#   feとfdは'Sigmoid'、'tanh'、'relu'、None 
#   以下の例はDecoderの活性化関数がNoneの場合
model = Autoencoder(inputs, hidden, fd=None)
```

### class Reconst

### def training_autoencoder

- 学習を行う関数
- fe, fdに引数を渡すことで活性化関数を設定可能
- ae_method="sparse"とすることでスパース化可能
    - rho, sの引数を渡すことでパラメータを変更可能

### def train_stacked

- Stacked Autoencoderの学習
    - 各層の学習は'training_autoencoder'を使用する
    - 各層の学習モデルはリストに格納して返す
- fe, fdの引数をリストで渡すことで層ごとに活性化関数を設定可能
    - 一括に適用したい場合はリストにしない
- ae_method="sparse"とすることでスパース化可能
    - rho, sの引数をリストで渡すことで層ごとにパラメータを適用可能
        - 一括に適用したい場合はリストにしない

