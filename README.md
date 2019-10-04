# autoencoder

## Overview
python_modules(private)内に存在していたautoencoderを切り離し整備しているところ

## Requirement

- python3
- numpy
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
from autoencoder import Reconst, Autoencoder, train_ae, train_stacked
```

違う階層に置く場合は適宜パスを通して使う


## src

オートエンコーダのモジュールと、使用例のファイルが入っています

- autoencoder.py
- sample.py

autoencoder.pyの中にはいくつかのクラスがあります

## output

samplei.pyを実行すると学習モデルファイルが生成されます

## src/autoencoder.py

オートエンコーダの定義、学習済みモデルを使った再構成、学習をするための関数が用意

## 
### class Autoencoder

```
# inputs: 入力サイズ
# hidden: 隠れ層サイズ
# fe:     Encoder(入力層→隠れ層)の活性化関数(Default is 'sigmoid')
# fd:     Decoder(隠れ層→出力層)の活性化関数(Default is 'sigmoid')
#   feとfdは'Sigmoid'、'tanh'、None 
#   以下の例はDecoderの活性化関数がNoneの場合
model = Autoencoder(inputs, hidden, fd=None)
```

### class Reconst

### def train_ae

学習を行う関数

### def train_stacked

Stacked Autoencoderの学習

各層の学習は'train_ae'を使用する

各層の学習モデルはリストに格納して返す


