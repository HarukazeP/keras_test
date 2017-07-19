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
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import re
import random
import sys
import datetime

path = get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
text = open(path).read().lower()
text = text.replace("\n", " ")
text = re.sub(r"[ ]+", " ", text)
text= text[0:10000]
#データ軽くするため

print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
print(chars)
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

def print_top5(list1):
    #与えられたリストから上位5件の値と対応する文字を表示
    dict_A = dict((i,c) for i,c in enumerate(list1))
    list_B = sorted(dict_A.items(), key=lambda x: x[1], reverse=True)
    list_C = list_B[0:5]
    for k,v in list_C:
        print (indices_char[k],v)
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

def sample_n_list(preds, num, temperature=1.0):
    #乱数でn個選ぶんでnp配列に格納
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    #print('\nafter_normalization\n')
    #print_top5(preds)
    probas = np.random.multinomial(1, preds, num)
    #print (probas,'\n\n')
    tmp_list=[]
    for i in range(0,num):
        #print (probas[i],'\n')
	#print(np.argmax(probas[i]),'\n')
        tmp_list.append(np.argmax(probas[i]))
    print(tmp_list)
    return tmp_list

# train the model, output generated text after each iteration
for iteration in range(1):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    today=datetime.datetime.today()
    print('date = ',today)
    model.fit(X, y,
              batch_size=128,
              epochs=1)

    start_index = random.randint(0, len(text) - maxlen - 1)

    #for diversity in [0.2, 0.5, 1.0, 1.2]:
    for diversity in [0.5, 1.0]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(1):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            print('\n\nbefore_sampling\n')
            print_top5(preds)
            next_index_list = sample_n_list(preds, 5, diversity)
            next_char_list=[]
            print('\nafter_sampling\n')
            for k in range(0, len(next_index_list)):
                next_char =indices_char[next_index_list[k]]
                next_char_list.append(next_char)
                print(k,':',next_char)
                
            
            # ↓とりあえずやってるだけ
            tmp_next_char =next_char_list[0]
            generated += tmp_next_char
            sentence = sentence[1:] + tmp_next_char
            
            #print('\nafter_sampling\n')
            #sys.stdout.write(tmp_next_char)
            sys.stdout.flush()
        print()
