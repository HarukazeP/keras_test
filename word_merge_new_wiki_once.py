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
path = './corpus/WikiSentWithEndMark1.txt'
text = open(path).read().lower()
text = text.replace("\n", " ")
text = re.sub(r"[^a-z ]", "", text)
text = re.sub(r"[ ]+", " ", text)
today=datetime.datetime.today()
print('read_end = ',today)

text_list=text.split(" ")
print('todal words:', len(text_list))
words = sorted(list(set(text_list)))
print('kind of words:', len(words))
word_indices = dict((c, i) for i, c in enumerate(words))
indices_word = dict((i, c) for i, c in enumerate(words))

# cut the text in semi-redundant sequences of maxlen_words characters
maxlen_words = 10
step = 3
f_sentences = []
r_sentences = []
next_words = []
for i in range(0, len(text_list) - maxlen_words*2 -1, step):
    f_sentences.append(text_list[i: i + maxlen_words])
    r_sentences.append(text_list[i + maxlen_words+1: i + maxlen_words+1+maxlen_words][::-1]) #逆順のリスト
    next_words.append(text_list[i + maxlen_words])
print('nb sequences:', len(f_sentences))

print('Vectorization...')
f_X = np.zeros((len(f_sentences), maxlen_words, len(words)), dtype=np.bool)
r_X = np.zeros((len(r_sentences), maxlen_words, len(words)), dtype=np.bool)
y = np.zeros((len(f_sentences), len(words)), dtype=np.bool)
for i, sentence in enumerate(f_sentences):
    for t, word in enumerate(sentence):
        f_X[i, t, word_indices[word]] = 1
    y[i, word_indices[next_words[i]]] = 1

for i, sentence in enumerate(r_sentences):
    for t, word in enumerate(sentence):
        r_X[i, t, word_indices[word]] = 1

# build the model: a single LSTM
print('Build model...')
f_input=Input(shape=(maxlen_words, len(words)))
f_layer=LSTM(128,)(f_input)

r_input=Input(shape=(maxlen_words, len(words)))
r_layer=LSTM(128,)(r_input)

merged_layer=add([f_layer, r_layer])

out_layer=Dense(len(words),activation='softmax')(merged_layer)

model=Model([f_input, r_input], out_layer)



optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

def print_top5(list1):
    #与えられたリストから上位5件の値と対応する文字を表示
    dict_A = dict((i,c) for i,c in enumerate(list1))
    list_B = sorted(dict_A.items(), key=lambda x: x[1], reverse=True)
    list_C = list_B[0:5]
    for k,v in list_C:
        print (indices_word[k],v)
    print()

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

# train the model, output generated text after each iteration
today=datetime.datetime.today()
print('fit_start = ',today)
model.fit([f_X,r_X], y,
              batch_size=128,
              epochs=1)
today=datetime.datetime.today()
print('fit_end = ',today)
model.save('word_merge_epoch1.h5')
today=datetime.datetime.today()
print('all_end = ',today)

