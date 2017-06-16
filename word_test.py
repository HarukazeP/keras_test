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
text = re.sub(r"[^a-z ]", "", text)
text = re.sub(r"[ ]+", " ", text)

text_list=text.split(" ")
print('todal words:', len(text_list))
words = sorted(list(set(text_list)) #まだここ変えたくらい
print('kind of words:', len(words))
#print(chars)
word_indices = dict((c, i) for i, c in enumerate(words))
indices_word = dict((i, c) for i, c in enumerate(words))

# cut the text in semi-redundant sequences of maxlen_words characters
maxlen_words = 10
step = 3
sentences = []
next_chars = []
for i in range(0, len(text_list) - maxlen_words, step):
    sentences = text_list[i: i + maxlen_words]
    next_chars.append(text_list[i + maxlen_words])
print('nb sequences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen_words, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, word_indices[char]] = 1
    y[i, word_indices[next_chars[i]]] = 1


# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen_words, len(chars))))
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
        print (indices_word[k],v)
    print()

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    print('\nafter_normalization\n')
    print_top5(preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# train the model, output generated text after each iteration
for iteration in range(2):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    today=datetime.datetime.today()
    print('date = ',today)
    model.fit(X, y,
              batch_size=128,
              epochs=1)

    start_index = random.randint(0, len(text_list) - maxlen_words - 1)

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = text_list[start_index: start_index + maxlen_words]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(2):
            x = np.zeros((1, maxlen_words, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, word_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            print('\n\nbefore_sampling\n')
            print_top5(preds)
            next_index = sample(preds, diversity)
            next_char = indices_word[next_index]
            
            generated += next_char
            sentence = sentence[1:] + next_char
            
            print('\nafter_sampling\n')
            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()
