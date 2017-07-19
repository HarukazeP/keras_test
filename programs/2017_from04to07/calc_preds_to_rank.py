# -*- coding: utf-8 -*-

from __future__ import with_statement
import datetime

today=datetime.datetime.today()
print('all_start = ',today)




#ランクリストと選択肢リストから，選択肢をランクリストの何番目に現れるか（順位）つきで並べた文字列を返す
#選択肢の語がランクリストにないときは-1
def search_rank(ra_list, ch_list):
    str_rank=''
    str_num=''
    k=0
    for x in ch_list:
        if x in ra_list:
            str_num=str(ra_list.index(x))
            str_rank=str_rank+x+': '+str_num+',  '
        else:
            #無いときは-1
            str_num='-1'
            str_rank=str_rank+x+': '+str_num+',  '
    #末尾のコンマとスペース消す
    k=len(str_rank)-3
    str_rank=str_rank[:k]
    
    return str_rank




file_rank='./rank_test_goudouzemi_2017_07_12_14_18_09.txt'
file_preds='./Result/preds_all_goudouzemi.txt'

file_out='./Result/preds_all_with_rank_goudouzemi.txt'

with open(file_rank, "r") as rank:
    with open(file_preds, "r") as preds:
        with open(file_out,"a") as out:
            for line in rank:
                rank_line=line.lower().replace('\n','').replace('\r','')
                preds_line=preds.readline().lower().replace('\n','').replace('\r','')
                
                rank_list=rank_line.split(' ')
                preds_list=preds_line.split(',  ')
                
                output=search_rank(rank_list, preds_list)
                out.write(output+'\n')



today=datetime.datetime.today()
print('all_end ',today)

