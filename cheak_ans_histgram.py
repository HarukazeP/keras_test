# -*- coding: utf-8 -*-

'''
min_modelをロードしてテストする用

python    : 2.7.12
keras     : 2.0.4
gensim    : 3.0.1
tensorflow: 1.1.0

プログラム全体の構成
    ・グローバル変数一覧
    ・関数群
    ・いわゆるmain部みたいなの

プログラム全体の流れ
    0.いろいろ前準備
    1.学習データの前処理
    2.fasttextのロードと辞書の作成
    3.モデルの定義
    4.モデルの学習
    5.val_loss最小モデルのロード
    6.テストの実行
    7.結果まとめの出力

'''

from __future__ import print_function
import numpy as np
import sys
import os
import os.path
import matplotlib.pyplot as plt



#----- いわゆるmain部みたいなの -----

# 0.いろいろ前準備
#開始時刻のプリント
print('start')

#モデルとか結果とかを格納するディレクトリの作成
argvs = sys.argv
argc = len(argvs)
if argc <1:    #rankNG,txtで1つ(python は含まれない)
    print('### ERROR: invalid argument! ###')
file_name=argvs[1] #ここ実行時の第二引数、〜〜.txtのファイル
#例： 2018_01_04_1450epoch100_e100_w10_add_bilstm_den1/rankNG.txt
save_path=file_name[:file_name.find('rankNG')]   #save_pathはmin_modelの手前の/まで
save_path=save_path+'NEW_TEST_'

rank_list=list()
i=0
with open(file_name, 'r') as f:
    for line in f:
        i+=1
        line=line[line.find('ans= ('):]
        tmp=line.split(', ')
        rank=tmp[1]
        rank_list.append(rank)

rank_array=np.array(rank_list,dtype=np.int)

plt.hist(rank_array, bins=10)
plt.savefig(save_path+'ans_histogram_10.png')
plt.hist(rank_array, bins=100)
plt.savefig(save_path+'ans_histogram_100.png')
print('save end')
