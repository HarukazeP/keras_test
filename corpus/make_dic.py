# -*- coding: utf-8 -*-


from __future__ import with_statement
import re
import sys
import datetime


today=datetime.datetime.today()
print('all_start = ',today)

path = './corpus/WikiSentWithEndMark1.txt'
all_words=[]
i=0
with open(path,'r') as f:
    for line in f:
        i+=1
        if(i%10000==0):
            print('line:', i)
        line=line.lower()
        line = line.replace("\n", " ")
        line = re.sub(r"[^a-z ]", "", line)
        line = re.sub(r"[ ]+", " ", line)
        line_list = line.split(" ")
        line_words=list(set(line_list))
        all_words.append(line_words)
        all_words=[x for i, x in enumerate(all_words) if i == all_words.index(x)]

today=datetime.datetime.today()
print('read_end = ',today)


all_words = sorted(all_words)
print('kind of words:', len(all_words))

with open('wiki_words.txt','r') as f2:
    for x in all_words:
        f2.write(str(x)+" ")

today=datetime.datetime.today()
print('all_end = ',today)

