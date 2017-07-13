# -*- coding: utf-8 -*-

from __future__ import with_statement
import datetime

today=datetime.datetime.today()
print('all_start = ',today)

ok_num=0
ng_num=0




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


#順位付き文字列と選択肢リストから最も順位の高い単語を返す
#どの語もランクリストにないときは#OTHERを返す
def serch_highest(str_rank, ch_list):
    tmp_list=str_rank.split(',  ')
    num_list=[]
    flag=0
    word=''
    for x in tmp_list:
        num=int(x[x.index(': ')+2:])
        num_list.append(num)
    min=max(num_list)+10
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



file_path='./Rank/rank_test_goudouzemi_2017_07_12_14_18_09.txt'
file_ok='./Result/rank_OK_goudouzemi.txt'
file_ng='./Result/rank_NG_goudouzemi.txt'
file_top10='./Result/rank_top10_goudouzemi.txt'
with open('./Ans/ans_goudouzemi_only.txt', "r") as ans:
    with open('./Ans/choices_goudouzemi_only.txt', "r") as choices:
        with open(file_path,"r") as rank:
            for line in rank:
                rank_line=line.lower().replace('\n','').replace('\r','')
                ans_line=ans.readline().lower().replace('\n','').replace('\r','')
                choices_line=choices.readline().lower().replace('\n','').replace('\r','')
                
                rank_list=rank_line.split(' ')
                choices_list=choices_line.split(' ')
                #TOP10を書き込み
                with open(file_top10,"a") as top:
                    for x in rank_list[:10] :
                        top.write(x+' ')
                    top.write('\n')
                
                output=search_rank(rank_list, choices_list)
                predict_word=serch_highest(output, choices_list)
                
                with open(file_ok,"a") as ok:
                    with open(file_ng,"a") as ng:
                        if ans_line==predict_word:    #正解したものについて各順位を書き込み
                            ok.write(output+'\n')
                            ok_num+=1
                        else:    #不正解のものついて各順位を書き込み
                            ng.write(output+'\n')
                            ng_num+=1
print('ok: ',ok_num)
print('ng: ',ng_num)
print('per:',1.0*ok_num/ng_num)

    
today=datetime.datetime.today()
print('all_end ',today)

