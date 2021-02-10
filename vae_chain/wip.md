vae_gaussian.py, vae_gaussian_ed.py, variational_autoencoder_ed.py, variational_autoencoder_v2.py における未修正点
1. set_optimizerメソッドが trainメソッドで呼ばれるようになっているため, プログラムを直接弄らないと自分で設定できない
   また, set_optimizerメソッドでto_gpuしているので, コンストラクトで呼び出すとどこかでto_cpuが必要になるなど修正が微妙に面倒.

2. 活性化関数が全層で共通になるようにしているので, 層ごとに設定できるようにしたい

3. _ed 版とそれ以外で weight_plot のencとdecの上下が逆

4. 正規化/標準化に使用した値をモデルに紐付けるできるようにしておいたほうがいいかもしれない

5. 上記全てのプログラムの親クラスを作りたい
