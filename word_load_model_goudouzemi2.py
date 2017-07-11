# -*- coding: utf-8 -*-

'''Example script to generate text from Nietzsche's writings.

At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
from keras.models import model_from_json
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input
from keras.layers import LSTM
from keras.layers import add
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras.utils.vis_utils import plot_model
import keras
import numpy as np
import re
import random
import sys
import datetime

today=datetime.datetime.today()
print('all_start = ',today)

maxlen_words = 10
#長さの閾値
th_len =maxlen_words/2

f_sentences = []
r_sentences = []
sents_num=0


#テストデータへ読み込みと前処理
#テストデータは1行1問で1行に<>が1つのみ
test_path = './corpus/center_goudouzemi.txt'
test_data = open(test_path).read().lower()
all_lines = test_data.split("\n")
for line in all_lines:
    tmp=line.split("<>")
    if(len(tmp)>1):
        f_tmp=re.sub(r"[^a-z ]", "", tmp[0])
        f_tmp = re.sub(r"[ ]+", " ", f_tmp)
        r_tmp=re.sub(r"[^a-z ]", "", tmp[1])
        r_tmp = re.sub(r"[ ]+", " ", r_tmp)            
        f_line=f_tmp.split(" ")
        r_line=r_tmp.split(" ")
        if (len(f_line)>th_len) and (len(r_line)>th_len):
            if (len(f_line)>maxlen_words):
                f_line=f_line[-1*maxlen_words:]
            if (len(r_line)>maxlen_words):
                r_line=r_line[:maxlen_words]    
            f_sentences.append(f_line)
            r_sentences.append(r_line[::-1])
            sents_num+=1
            with open('testdata_goudouzemi.txt', 'a') as data:
                data.write(line+'\n')

print('test_sentences:', sents_num)

#辞書の作成
path = './corpus/miniWiki_tmp8.txt'
text = open(path).read().lower()
text = text.replace("\n", " ")
text = re.sub(r"[^a-z ]", "", text)
text = re.sub(r"[ ]+", " ", text)
text_list=text.split(" ")
len_text=len(text_list)
words = sorted(list(set(text_list)))
words.append("#OTHER")
len_words=len(words)
word_indices = dict((c, i) for i, c in enumerate(words))
indices_word = dict((i, c) for i, c in enumerate(words))
today=datetime.datetime.today()
print('dict_end = ',today)


#モデルをロード
print('Load model...')
model_file='word_merge_wiki_epoch1_tmp8_2017_07_11_14_11_55.json'
weight_file='word_merge_wiki_epoch1_tmp8_2017_07_11_14_11_55.h5'

json_string = open(model_file).read()
model = model_from_json(json_string)
model.load_weights(weight_file)

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


def print_rank(list1, filename):
    #与えられたリストからランキング順にファイルへ書き込み
    print('Write rank ...')
    dict_A = dict((i,c) for i,c in enumerate(list1))
    list_B = sorted(dict_A.items(), key=lambda x: x[1], reverse=True)
    with open("filename", "a") as file:
        for k,v in list_B:
            str=indices_word[k]+ ' '
            file.write(str)
        file.write('\n')

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    #print('\nafter_normalization\n')
    #print_top5(preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
    
def search_word_indices(word):
    if word in word_indices:
        return word_indices[word]
    else:
        return word_indices["#OTHER"]

#テストの実行
print('Test starts ...')
today=datetime.datetime.today()
print('date = ',today)
for i in range(sents_num):
    f_x = np.zeros((1, maxlen_words, len_words))
    r_x = np.zeros((1, maxlen_words, len_words))
    for t, word in enumerate(f_sentences[i]):
        f_x[0, t, search_word_indices(word)] = 1.
    for t, word in enumerate(r_sentences[i]):
        r_x[0, t, search_word_indices(word)] = 1.
    preds = model.predict([f_x,r_x], verbose=0)[0]
    #print('\n\nbefore_sampling\n')
    today=datetime.datetime.today()
    today_str = today.strftime("%Y_%m_%d_%H_%M_%S")
    filename='test_goudouzemi_'+today_str
    print_rank(preds, 'rank_'+filename+'.txt')
    next_index = sample(preds)
    next_word = indices_word[next_index]
    with open('preds_'+filename+'.txt', "a") as file:
        file.write(next_word+'\n')
    
today=datetime.datetime.today()
print('all_end ',today)

