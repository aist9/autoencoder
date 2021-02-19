# variational_autoencoder

## usage
とりあえず使うなら以下のようにする( *argsはそれぞれ適切なものを指定する )
```
vae = VAE(*args)

vae.train(*args)

feat, reconst, error = vae.reconst(*args)
```

## インスタンス生成
```
vae = VAE(input_shape, hidden, act_func, out_func ,use_BN, init_method, folder, is_gauss_dist)
```
- input_shape ... 入力データの次元数. 整数型.
- hidden ... 中間層のパーセプトロン数, [h1, h2, ..., hn]のようにリストで渡す
- act_func ... 活性化関数, strか関数を渡す.　基本的には'tanh'を指定する
- out_func ... 出力層の関数, strか関数を渡す. 基本的には'sigmoid'を指定する. gauss版のときはvarのみにかかる.
- use_BN ... BatchNornalizationを使用するか否か. Bool型.
- init_method ... レイヤーの重みの初期化方法. strか関数を渡す. 無指定がおすすめ 'xavier'が入る.
- folder ... モデルの保存先のフォルダ. strで渡す. 実際には folder+"{}".format( input_shape + hidden ) みたいなディレクトリに保存される.
- is_gauss_dist ... 出力をガウス分布を仮定したものにするか. Bool型. デコーダの出力が2出力になる. 基本的にはFalse.

## 学習
```
vae.train(train,epoch,batch,C=1.0,k=1, gpu_num=0,valid=None, is_plot_weight=False)
```
- train ... 学習データ. ndarray
- epoch ... エポック数. 整数型
- batch ... ミニバッチサイズ. 整数型
- C ... latent loss の係数. 基本的に1. float型
- k ... iterateあたりのreconst lossの計算回数. 基本的に1. 整数型
- gpu_num ... 使用するGPUの番号. 基本的に0. 使用しないなら-1を選択. 整数型
- valid ... validation に使用するデータ. 指定するとvalidationデータの平均MSEが出力されるようになる. ndarray
- is_plot_weight ... レイヤーの重みの変動を記録するか否か. モデルの保存フォルダに出力. Bool型

## 結果の出力
```
feat, reconst, error = vae.reconst(data)
```
- data ... テストしたいデータ. ndarray
- feat ... 潜在空間の出力.（encoderの出力） ndarray
- reconst ... 再構成データ. ndarray
- error ... reconst と data のMSE. ndarray

## その他使い方
基本的には sampleコードを参照.

### 学習済みモデルのロード
pathには学習済みモデルのディレクトリを入れる. ロードされるモデルは path+'/model.npz' となる. Noneならインスタンス生成時のパスが使用される<br>
```
vae.load(path=None)
```

### 再学習
```
vae.load(path=None)
vae.train(*args)
```

## 実装についての薀蓄

### stacked aeではない
ae_chainのものと異なり, 層ごとにモデルを構築していない. これに関しては reparameterization trick のことを考えるとやらないほうが良い<br>
層ごとに活性化関数を変更しない. 出力層は除く. やろうと思えば実装できるがやっていない.

### ファイル名の _edについて
エンコーダとデコーダが分かれていることを示す<br>
optimizerが2つ作られるので無印版と比べ学習に影響があると思われる<br>
現状ではoptimizerのパラメータは共通するように実装しているが, インスタンス生成後に以下のように再代入すれば個別に設定できる<br>
'''
vae.enc_opt = hoge1
vae.dec_opt = hoge2
vae.enc_opt.setup(self.encoder)
vae.dec_opt.setup(self.decoder)
'''

### is_gauss_dist=True のバージョンについて
デコーダがmuとvarの2出力であるVAE<br>
出力にGauss分布を仮定したもので, 元論文にも書いてある. ただし, VAEで検索すると1出力のほうがよく見つかる<br>
1出力(bernoulli分布を仮定したバージョン)のものと以下の点が異なる<br>
- ロス関数に gaussian_nll(negative log likelihood) を使う. 1出力verでは bernoulli_nll.
- 入力データを0~1にする必要がない. ->  bernoulli_nll では第二引数(decoder output)に sigmoid が適用されるため正規化が必須であるが, こちらではそれがない.
- ロスがマイナスの値になる. おそらく正常な動作だが, -infに発散することがある.
- varにsigmoidをかけることで防げるが, これが正しい実装なのかよくわからない. 詳しくは後述.
- 実装したはいいが, 再構成データがどれに相当するかわからない（おそらくデコーダのmu?）
これを実装した最大の理由は「非正則化異常度を用いた工業製品の異常検知」の手法を実験するため<br>
論文中の D_VAE/A_VAE/M_VAEを計算するため, ロス関数をgaussian_nllではなく直接実装している(結果が変わらないことは確認済み)<br>
-> が, 結局printしていないのでgaussian_nllに変えてもいいかも<br>
VAEクラスのreconstメソッドの引数 "unregular" をTrueにするとMSEの代わりに上記手法の誤差(M_VAE)を出力する<br>

### gaussバージョンでのdecoder出力における活性化関数の有無について
MNISTで実験, sigmoidでのみ試行<br>
bernoulli分布と異なり, gaussian分布では平均・分散が0~1である必要はないはずだが, MNISTで試行するとsigmoidをかけたほうが学習がうまく行く<br>
- dec_mu, dec_var ともに無し... ロスが負の方向に増大していく. 潜在特徴や再構成の結果は上手くいかない.
- dec_mu, dec_var ともに有り... ロスは正の方向に減少していく. 潜在特徴でクラス判別可. 再構成可.
- dec_mu  のみ有り          ... ロスは正の方向に増大していく. nanが出るため学習不可.
- dec_var のみ有り          ... ロスは正の方向に減少していく. 潜在特徴でクラス判別可. 再構成可.
結論 : 入力を標準化(平均0分散1)した結果, dec_varにsigmoidをかけた場合のみ正常に機能したためこれを標準実装とする(eluやsoftplusでも良さそう？)<br>
dec_ln_varとして学習できていない？ ln_varとして学習できているなら負の値を許容できるはずだが, sigmoidで負の値を弾いているから学習が上手くいっている気がする<br>
MNISTでは上手くいっただけで, ものによっては活性化関数がないほうがいいかもしれない. out_funcで設定できるようにした<br>




