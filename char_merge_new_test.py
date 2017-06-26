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
from keras.layers import Add
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras.utils.vis_utils import plot_model
import keras
import numpy as np
import re
import random
import sys
import datetime

path = get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
text = open(path).read().lower()
text = text.replace("\n", " ")
text = re.sub(r"[^a-z ]", "", text)
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
f_sentences = []
r_sentences = []
next_chars = []

for i in range(0, len(text) - maxlen*2 -1, step):
    f_sentences.append(text[i: i + maxlen])
    r_sentences.append(text[i + maxlen+1: i + maxlen+1+maxlen][::-1]) #逆順の文字列
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(f_sentences))

print('Vectorization...')
f_X = np.zeros((len(f_sentences), maxlen, len(chars)), dtype=np.bool)
r_X = np.zeros((len(r_sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(f_sentences), len(chars)), dtype=np.bool)
for i, f_sentence in enumerate(f_sentences):
    for t, char in enumerate(f_sentence):
        f_X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

for i, r_sentence in enumerate(r_sentences): #ここもう少しうまい書き方ありそう
    for t, char in enumerate(r_sentence):
        r_X[i, t, char_indices[char]] = 1


# build the model: a single LSTM
# ここ参考にした
# https://stackoverflow.com/questions/43196636/how-to-concatenate-two-layers-in-keras
# https://stackoverflow.com/questions/44042173/concatenate-merge-layer-keras-with-tensorflow
print('Build model...')
'''
#model = Sequential()
#model.add(LSTM(128, input_shape=(maxlen, len(chars))))
#model=add([f_X, r_X])
#model.add(Dense(len(chars)))
#model.add(Activation('softmax'))


forward_model = Sequential()
forward_model.add(LSTM(128, input_shape=(maxlen, len(chars))))
reverse_model = Sequential()
reverse_model.add(LSTM(128, input_shape=(maxlen, len(chars))))

merged = Merge([forward_model, reverse_model], mode='sum')

model = Sequential()
model.add(merged)
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

'''
f_input=Input(shape=(maxlen, len(chars)))
f_layer=LSTM(128,)(f_input)

r_input=Input(shape=(maxlen, len(chars)))
r_layer=LSTM(128,)(r_input)

merged_layer=Add()([f_layer.output, f_layer.output])

out_layer=Dense(len(chars),activation='softmax')(merged_layer)

model=Model([f_layer.input, r_layer.input], out_layer)

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

# train the model, output generated text after each iteration
for iteration in range(1):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    today=datetime.datetime.today()
    print('date = ',today)
    model.fit([f_X, r_X], y, 
              batch_size=128,
              epochs=1)

    start_index = random.randint(0, len(text) - maxlen -maxlen - 1)

    #for diversity in [0.2, 0.5, 1.0, 1.2]:
    for diversity in [0.5, 1.0]:
    
        print()
        print('----- diversity:', diversity)

        generated = ''
        f_sent = text[start_index: start_index + maxlen]
        r_sent = text[start_index + maxlen+1 : start_index + maxlen+1 +maxlen][::-1]  #逆順の文字列
        generated += f_sent
        print('----- Generating with seed: "' + f_sent + '"')
        sys.stdout.write(generated)
        
        flag=0
        for i in range(1):
            f_x = np.zeros((1, maxlen, len(chars)))
            r_x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(f_sent):
                f_x[0, t, char_indices[char]] = 1.
            for t, char in enumerate(r_sent):  #ここもう少しうまい書き方ありそう
                r_x[0, t, char_indices[char]] = 1.

            preds = model.predict([f_x, r_x], verbose=0)[0]
            #print('\n\nbefore_sampling\n')
            #print_top5(preds)
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]
            generated += next_char
            f_sent = f_sent[1:] + next_char
            #この方法ではr_sentが出せない
            #実際は固定されたテストデータだろうからこのプログラムのようなことは起きないけれども

            
            print('\nafter_sampling\n↓')
            sys.stdout.write(next_char)
            sys.stdout.flush()
            print('\n↑')
        print()
plot_model(model, to_file='model_char_merge.png')
