# my default import modules
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# **********************************************
# Viasualize
# **********************************************

# weight visualize
def weight_plot(
        weight, plot_num,
        reshape_size=None, save_path='w.png'):

    # 自動でsubplotの分割数を決める
    row = int(np.sqrt(plot_num))
    mod = plot_num % row
    
    # 保存形式が.pdfの場合の処理
    if save_path in '.pdf':
        pp = PdfPages(save_path)

    # plot
    for i in range(plot_num):
        # 次の層i番目に向かう重みの抜き出し
        w = weight[i]
        # reshape(指定があれば)
        if reshape_size is not None:
            w = w.reshape(reshape_size)
        # 自動でsubplotの番号を与えplot
        plt.subplot(row, row+mod, 1+i)
        plt.imshow(w, cmap='gray')
    
    # 保存処理
    if save_path in '.pdf':
        pp.savefig()
        plt.close()
        pp.close()
    else:
        plt.savefig(save_path)
        plt.close()

# Bias visualize
def bias_plot(
        bias, reshape_size=None, save_path='b.png'):

    # reshape
    if reshape_size is not None:
        bias = bias.reshape(reshape_size)

    # 保存形式が.pdfの場合の処理
    if save_path in '.pdf':
        pp = PdfPages(save_path)
    # plot
    plt.imshow(bias, cmap='gray')

    # 保存処理
    if save_path in '.pdf':
        pp.savefig()
        plt.close()
        pp.close()
    else:
        plt.savefig(save_path)
        plt.close()

