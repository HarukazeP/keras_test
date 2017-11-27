# -*- coding: utf-8 -*-

'''
fasttextのベクトルをファイル出力
サーバでgensimのインストールうまくいかないその対策
'''

from __future__ import print_function
from gensim.models.wrappers import FastText
import keras
import numpy as np
import re
import random
import sys
import datetime
import gensim
import os

vec_size=100

start_time=datetime.datetime.today()
print('all_start = ',start_time)
today_str = start_time.strftime("%Y_%m_%d_%H%M")

today_str=today_str+'_vec'
os.mkdir(today_str)
today_str='./'+today_str+'/'
#日付名+epoch数のフォルダを作成し、結果はすべてそこに格納



#時間表示
def print_time(str1):
    today=datetime.datetime.today()
    print(str1)
    print(today)



def printvec(train_path, vec_path):
	#1.小文字化など前処理したファイルを作成
	print('\nPreprpcessing training data...')
	tmp_path=train_path[:-4]+'_cleaned.txt'
	with open(train_path) as f_in:
	    with open(tmp_path, 'w') as f_out:
	        for line in f_in:
	            text=line.lower()
	            text = re.sub(r"[^a-z ]", "", text)
	            text = re.sub(r"[ ]+", " ", text)
	            f_out.write(text)
	train_path=tmp_path
	
	#2.辞書の作成
	print('\nMake dic...')
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
	len_words=len(words)
	word_indices = dict((c, i+1) for i, c in enumerate(words))
	indices_word = dict((i+1, c) for i, c in enumerate(words))
	# 0番目はパディング用の数字なので使わないことに注意
	
	#3.fasttextの学習
	myft_path='/home/tamaki/M1/Keras/mine2017_8to11/fastText/fasttext'
	ft_model = FastText.train(ft_path=myft_path, corpus_file=train_path, size=vec_size, window=5, min_count=0)
	ft_model.save(today_str+'ft.model')

	#4.ベクトルのファイル出力
	with open(vec_path, 'w') as file:
		for i in range(len_words):
		    if i!=0:
		    	word=indices_word[i]
		    	if word in ft_model.wv.vocab:
		    		vec=ft_model[word]
		    	else:
		    		vec=np.zeros((vec_size),dtype=np.float32)
		    	output=word+' > 'str(vec)+'\n'
		    	file.write(output)
		    	
	#5.モデルをリセット
	ft_model.reset_weights()




#読み込むもの

train_big='../corpus/WikiSentWithEndMark1.txt'   # 約5.8GB，約2000万行
train_mid='../corpus/miniWiki_tmp8.txt'   # 約1.5MB，約5000行
train_small='../corpus/nietzsche.txt'   # 約600KB，約1万行

train_text8='../corpus/text8.txt'   # 約95MB 1行のみ　http://mattmahoney.net/dc/text8.zip


#出力されるもの

vec_big=today_str+'vec_Wiki.txt'
vec_mid=today_str+'vec_miniWiki.txt'
vec_small=today_str+'vec_nietzsche.txt'
vec_text8=today_str+'vec_text8.txt'


print_vec(train_big, vec_big, model1)
print_time('big end')

print_vec(train_text8, vec_text8, model2)
print_time('text8 end')

print_vec(train_mid, vec_mid, model3)
print_time('mif end')

print_vec(train_small, vec_small, model4)
print_time('small end')

end_time=datetime.datetime.today()
print('all_end = ',end_time)
diff_time=end_time-start_time
print('total =',diff_time)
