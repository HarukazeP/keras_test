# -*- coding: utf-8 -*-

'''
fasttextでベクトルにして学習するモデル
embeddingレイヤーでfasttextのベクトルを利用
多クラス分類みたいな感じで各単語の確率分布を出力するモデル

python    : 2.7.12
keras     : 2.0.4
gensim    : 3.0.1
tensorflow: 1.1.0

プログラム全体の構成
    グローバル変数一覧
    関数群
    いわゆるmain部みたいなの

プログラム全体の流れ
    1.学習データの前処理
    2.fasttextのロード
    3.モデルの定義
    4.モデルの学習
    5.val_loss最小モデルのロード
    6.テストの実行

'''

from __future__ import print_function
from keras.models import Sequential, Model
from keras.models import model_from_json
from keras.layers import Dense, Activation, Input, Embedding
from keras.layers import LSTM
from keras.layers import add
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras.utils.vis_utils import plot_model
from gensim.models.wrappers import FastText
import keras
import numpy as np
import re
import random
import sys
import datetime
import gensim
import os
import matplotlib.pyplot as plt

#----- グローバル変数一覧 -----
my_epoch=100
vec_size=100
maxlen_words = 10

KeyError_set=set() #TODO これグローバル変数に必要？

# TODO このあたりの変数，グローバル変数にするかどうか確認
# step = 3
# today_str #TODO today_strを使わずに関数の引数はfpathにするとか
# len_words
# word_indices
# indices_word


#TODO 最後にタブをスペースに置換



#----- 関数群 -----
#TODO 使う順に並び替え

#時間表示
def print_time(str1):
    today=datetime.datetime.today()
    print(str1)
    print(today)

#loss, val_lossの追加更新
def conect_hist(list_loss, list_val_loss, new_history):
    list_loss.extend(new_history.history['loss'])
    list_val_loss.extend(new_history.history['val_loss'])


# 損失の履歴をプロット
def plot_loss(list_loss, list_val_loss, title='model loss'):
    plt.plot(list_loss, color="blue", marker="o", label='loss')
    plt.plot(list_val_loss, color="green", marker="o", label='val_loss')
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(today_str+'loss_graph.png')
    #plt.show()は行うとプログラム中断されるからNG

#単語から辞書IDを返す
#TODO ここword_indicesをグローバル変数にしないように変更
def search_word_indices(word):
    if word in word_indices:
        return word_indices[word]
    else:
        return word_indices["#OTHER"]


#fasttextのベクトルを得る
#未知語の場合には[0,0,0, ... ,0]みたいなやつにとりあえずしてる
#未知語は集合に格納し，あとでファイル出力
#TODO ft_modelも引数にする
#TODO KeyErrorも引数に？
def get_ft_vec(word):
    if word in ft_model.wv.vocab:
        return ft_model[word]
    else:
        KeyError_set.add(word)    #要素を追加
        return np.zeros((vec_size),dtype=np.float32)


#与えられた確率付き単語リストからランキング順に単語のみファイルへ書き込み
def print_rank(list1, fname):
    dict_A = dict((i,c) for i,c in enumerate(list1))
    list_B = sorted(dict_A.items(), key=lambda x: x[1], reverse=True)
    with open(fname, "a") as file:
        #print('Write rank ...')
        for k,v in list_B:
            if k!=0:
                #idが0は存在しないので飛ばす
                str=indices_word[k]+ " ### "
                file.write(str)
        file.write('\n')


#単語とランクリストから単語の順位をstring型で返す
def word_to_rank(word, ra_list):
    str_num=''
    if word in ra_list:
        str_num=str(ra_list.index(word))
    else:
        #無いときは-1
        str_num='-1'

    return str_num
    


#ランクリストと選択肢リストから，選択肢をランクリストの何番目に現れるか（順位）つきで並べた文字列を返す
#選択肢の語がランクリストにないときは-1
def search_rank(ra_list, ch_list):
    str_rank=''
    str_num=''
    k=0
    for x in ch_list:
        str_num=word_to_rank(x, ra_list)
        str_rank=str_rank+x+': '+str_num+' ### '
    #末尾のシャープとか消す
    k=len(str_rank)-5
    str_rank=str_rank[:k]

    return str_rank



#順位付き文字列と選択肢リストから最も順位の高い単語を返す
#どの語もランクリストにないときは#OTHERを返す
def serch_highest(str_rank, ch_list):
    tmp_list=str_rank.split(' ### ')
    num_list=[]
    flag=0
    word=''
    for x in tmp_list:
        num=int(x[x.index(': ')+2:])
        num_list.append(num)
    min=max(num_list)+10    #この10に特に意味はない．単に大きい数字にしたいだけ
    min_ct=0
    ct=0
    for i in num_list:
        if (i>=0) and (min>i):
            flag=1
            min=i
            min_ct=ct
        ct+=1

    if flag==0:
        word='#OTHER'
    else:
        word=ch_list[min_ct]

    return word


#確率で選んだ1語の正誤をファイル書き込み，正誤結果を返す
def calc_rank1word(pred, ans, list_rank):
    rank_pred=word_to_rank(pred, list_rank)
    rank_ans=word_to_rank(ans, list_rank)
    out=''
    with open(today_str+'rankOK.txt', 'a') as rOK:
        with open(today_str+'rankNG.txt', 'a') as rNG:
            out='pred= '+pred+' : '+rank_pred+'     '+'ans= '+ans+' : '+rank_ans+'\n'
            if pred==ans:
                rOK.write(out)
                OK_num=1
            else:
                rNG.write(out)
                OK_num=0
    return OK_num



#確率で選択肢から選んだ際の正誤をファイル書き込み，正誤結果を返す
def calc_rank4choices(choices, ans, list_rank):
    #まず選択肢をリストへ
    choi_list=choices.split(' ### ')
    out=search_rank(list_rank, choi_list)
    pred=serch_highest(out, choi_list)

    with open(today_str+'choicesOK.txt', 'a') as cOK:
        with open(today_str+'choicesNG.txt', 'a') as cNG:
            out=out+'\n'
            if pred==ans:
                cOK.write(out)
                OK_num=1
            else:
                cNG.write(out)
                OK_num=0
    return OK_num



#単語一覧リストの作成
def make_dic(train_path):
	s=set()
	with open(train_path) as f:
	    for line in f:
	        text=line.lower()
	        text = text.replace("\n", " ").replace('\r','')
	        text = re.sub(r"[ ]+", " ", text)
	        text_list=text.split(" ")
	        tmp_set=set(text_list)
	        s.update(tmp_set)

	words = sorted(list(s))
	words.append("#OTHER")
	return words



# モデルの構築
def build_model():	#TODO この関数の作成，引数どうする？
	print('Build model...')
	f_input=Input(shape=(maxlen_words,))
	f_emb=Embedding(output_dim=vec_size, input_dim=len_words+1, input_length=maxlen_words, mask_zero=True, weights=[embedding_matrix], trainable=False)(f_input)

	f_layer=LSTM(128)(f_emb)

	r_input=Input(shape=(maxlen_words,))
	r_emb=Embedding(output_dim=vec_size, input_dim=len_words+1, input_length=maxlen_words, mask_zero=True, weights=[embedding_matrix], trainable=False)(r_input)
	r_layer=LSTM(128)(r_emb)

	merged_layer=add([f_layer, r_layer])

	out_layer=Dense(len_words,activation='softmax')(merged_layer)

	model=Model([f_input, r_input], out_layer)

	optimizer = RMSprop()
	model.compile(loss='categorical_crossentropy', optimizer=optimizer)

	return model










































#----- いわゆるmain部みたいなの -----

# 0.いろいろ前準備

#開始時刻のプリント
start_time=datetime.datetime.today()
print('all_start = ',start_time)
today_str = start_time.strftime("%Y_%m_%d_%H%M")

#モデルとか結果とかを格納するディレクトリの作成
today_str=today_str+'epoch'+str(my_epoch)
if os.path.exists(today_str)==False:
	os.mkdir(today_str)
today_str='./'+today_str+'/'

#学習データの候補
train_big='../corpus/WikiSentWithEndMark1.txt'   # 約5.8GB，約2000万行
train_mid='../corpus/miniWiki_tmp8.txt'   # 約1.5MB，約5000行
train_small='../corpus/nietzsche.txt'   # 約600KB，約1万行

train_text8='../corpus/text8.txt'   # 約95MB 1行のみ, wcで数えたら約1700万単語  http://mattmahoney.net/dc/text8.zip





# 1.学習データの前処理など
tmp_path = train_text8        #使用する学習データ
print('Loading  '+tmp_path)
train_path=preprocess(tmp_path)	#TODO この関数の作成

#単語辞書というか索引の作成
words=make_dic(train_path)	#TODO この関数の作成
len_words=len(words)
print('kind of words:', len_words)
word_indices = dict((c, i+1) for i, c in enumerate(words))
indices_word = dict((i+1, c) for i, c in enumerate(words))
# ↑ 0番目はパディング用の数字なので使わないことに注意

print_time('make dic end')

# 2.fasttextのロードなど

#TODO ここロードの処理書く
'''
ちなみに学習はこれ
#fasttextの学習


print('Learning fasttext...')

myft_path='/home/tamaki/M1/Keras/mine2017_8to11/fastText/fasttext'
ft_model = FastText.train(ft_path=myft_path, corpus_file=train_path, size=vec_size, window=5, min_count=0)
ft_model.save(today_str+'ft.model')
# FastTextはcbowとskipgramの二つの学習方法があるがデフォルトではcbow
#ここsaveに少し時間かかる

'''


#embeddingで用いる，単語から行列への変換行列
embedding_matrix = np.zeros((len_words+1, vec_size))
for i in range(len_words):
    if i!=0:
        embedding_matrix[i] = get_ft_vec(indices_word[i])
        #IDが0の単語が存在しないので0は飛ばす


print_time('FastText end')

# 3.モデルの定義
my_model=build_model():	#TODO この関数の作成，引数どうする？


# 4.モデルの学習
list_loss=list()
list_val_loss=list()
min_i=0
for ep_i in range(my_epoch):
	print('\nEPOCH='+str(ep_i+1)+'/'+str(my_epoch)+'\n')
	loss, val_loss=model_fit(train_path, my_model)	#TODO この関数の作成
	#TODO returnの仕方も要検討







print_time('fit end')

# 5.val_loss最小モデルのロード



# 6.テストの実行
test_path = '../corpus/tmp_testdata_after.txt'     #答えつきテストデータ
ch_path= '../corpus/tmp_choices_after.txt'     #選択肢つきテストデータ


















