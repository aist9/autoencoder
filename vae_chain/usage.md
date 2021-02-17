# variational_autoencoder

## usage
とりあえず使うなら以下のようにする.( *argsはそれぞれ適切なものを指定する )

vae = VAE(*args)

vae.train(*args)

feat, reconst, error = vae.reconst(*args)

## インスタンス生成
vae = VAE(input_shape, hidden, act_func, out_func ,use_BN, init_method, folder, is_gauss_dist)
- input_shape ... 入力データの次元数. 整数型.
- hidden ... 中間層のパーセプトロン数, [h1, h2, ..., hn]のようにリストで渡す
- act_func ... 活性化関数, strか関数を渡す.　基本的には'tanh'を指定する
- out_func ... 出力層の関数, strか関数を渡す. 基本的には'sigmoid'を指定する. gauss版のときはvarのみにかかる.
- use_BN ... BatchNornalizationを使用するか否か. Bool型.
- init_method ... レイヤーの重みの初期化方法. strか関数を渡す. 無指定がおすすめ 'xavier'が入る.
- folder ... モデルの保存先のフォルダ. strで渡す. 実際には folder+"{}".format( input_shape + hidden ) みたいなディレクトリに保存される.
- is_gauss_dist ... 出力をガウス分布を仮定したものにするか. Bool型. デコーダの出力が2出力になる. 基本的にはFalse.

## 学習
vae.train(train,epoch,batch,C=1.0,k=1, gpu_num=0,valid=None, is_plot_weight=False)
- train ... 学習データ. ndarray
- epoch ... エポック数. 整数型
- batch ... ミニバッチサイズ. 整数型
- C ... latent loss の係数. 基本的に1. float型
- k ... iterateあたりのreconst lossの計算回数. 基本的に1. 整数型
- gpu_num ... 使用するGPUの番号. 基本的に0. 使用しないなら-1を選択. 整数型
- valid ... validation に使用するデータ. 指定するとvalidationデータの平均MSEが出力されるようになる. ndarray
- is_plot_weight ... レイヤーの重みの変動を記録するか否か. モデルの保存フォルダに出力. Bool型

## 結果の出力
feat, reconst, error = vae.reconst(data)
- data ... テストしたいデータ. ndarray
- feat ... 潜在空間の出力.（encoderの出力） ndarray
- reconst ... 再構成データ. ndarray
- error ... reconst と data のMSE. ndarray

## その他使い方
基本的には sampleコードを参照.

### 学習済みモデルのロード
pathには学習済みモデルのディレクトリを入れる. ロードされるモデルは path+'/model.npz' となる. Noneならインスタンス生成時のパスが使用される.

vae.load(path=None)

### 再学習
vae.load(path=None)

vae.train(*args)

