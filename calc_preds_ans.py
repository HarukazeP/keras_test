# -*- coding: utf-8 -*-

from __future__ import with_statement
import datetime

today=datetime.datetime.today()
print('all_start = ',today)

ok_num=0
ng_num=0
for i in range(10):
    str_i=str(i)
    file_path='./Preds/preds_test_goudouzemi_'+str_i+'.txt'
    file_ok='./Result/preds_OK_goudouzemi_'+str_i+'.txt'
    file_ng='./Result/preds_NG_goudouzemi_'+str_i+'.txt'
    with open('./Ans/ans_goudouzemi_only.txt', "r") as ans:
        with open(file_path,"r") as preds:
            for line in preds:
                preds_line=line.lower().replace('\n','').replace('\r','')
                ans_line=ans.readline().lower().replace('\n','').replace('\r','')
                with open(file_ok,"a") as ok:
                    with open(file_ng,"a") as ng:
                        if ans_line==preds_line:
                            print(i,ans_line,preds_line)
                            ok.write(ans_line)
                            ok_num+=1
                        else:
                            st='ans:'+ans_line+'   preds:'+preds_line
                            st=st+'\n'
                            ng.write(st)
                            ng_num+=1
print('ok: ',ok_num)
print('ng: ',ng_num)
print('per:',1.0*ok_num/ng_num)

    
today=datetime.datetime.today()
print('all_end ',today)

