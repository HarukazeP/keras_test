# -*- coding: utf-8 -*-

'''
min_modelをロードしてテストする用

python    : 2.7.12
keras     : 2.0.4
gensim    : 3.0.1
tensorflow: 1.1.0

プログラム全体の構成
    ・グローバル変数一覧
    ・関数群
    ・いわゆるmain部みたいなの

プログラム全体の流れ
    0.いろいろ前準備
    1.学習データの前処理
    2.fasttextのロードと辞書の作成
    3.モデルの定義
    4.モデルの学習
    5.val_loss最小モデルのロード
    6.テストの実行
    7.結果まとめの出力

'''

from __future__ import print_function
from keras.models import Sequential, Model
from keras.models import model_from_json
from keras.layers import Dense, Activation, Input, Embedding
from keras.layers import LSTM, Bidirectional
from keras.layers import add
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import keras
import numpy as np
import re
import sys
import datetime
import os
import os.path
import subprocess

#----- グローバル変数一覧 -----
my_epoch=100
vec_size=100
maxlen_words = 10   #仮, main部で上書き？
KeyError_set=set()
save_path=''
tmp_vec_dict=dict()


#----- 関数群 -----

#時間表示
def print_time(str1):
    today=datetime.datetime.today()
    print(str1)
    print(today)
    return today

#学習データやテストデータへの前処理
def preprocess_line(before_line):
    after_line=before_line.lower()
    after_line=after_line.replace('0', ' zero ')
    after_line=after_line.replace('1', ' one ')
    after_line=after_line.replace('2', ' two ')
    after_line=after_line.replace('3', ' three ')
    after_line=after_line.replace('4', ' four ')
    after_line=after_line.replace('5', ' five ')
    after_line=after_line.replace('6', ' six ')
    after_line=after_line.replace('7', ' seven ')
    after_line=after_line.replace('8', ' eight ')
    after_line=after_line.replace('9', ' nine ')
    after_line = re.sub(r'[^a-z]', ' ', after_line)
    after_line = re.sub(r'[ ]+', ' ', after_line)

    return after_line


#listの各要素を単語で連結してstring型で返す
def list_to_sent(list_line, start, end):
    sent=' '.join(list_line[start:end])
    return sent


#学習データへの前処理を行う
#小文字化，アルファベット以外の文字の削除，1万単語ごとに分割
def preprocess(train_path):
    max_len=50000
    new_path=train_path[:-4]+'_preprpcessed'+str(max_len)+'.txt'
    if os.path.exists(new_path)==False:

        print('Preprpcessing training data...')
        text=''
        text_len=0
        i=0
        with open(train_path) as f_in:
            with open(new_path, 'w') as f_out:
                for line in f_in:
                    #この前処理はtext8とかの前処理と同じ
                    line=preprocess_line(line)
                    line_list=line.split(' ')
                    line_len=len(line_list)
                    #max_len以下の時は連結して次へ
                    if(text_len+line_len <= max_len):
                        if(text_len==0):
                            text=line
                        else:
                            text=text+' '+line
                        text_len=text_len+line_len
                    #max_lenより長いときはmax_len単語ごとに区切ってファイルへ書き込み
                    else:
                        while (line_len>max_len):
                            if(text_len==0):
                                text=list_to_sent(line_list,0,max_len)
                            else:
                                text=text+' '+list_to_sent(line_list,0,max_len-text_len)
                            f_out.write(text+'\n')
                            text=''
                            text_len=0
                            #残りの更新
                            line_list=line_list[max_len-text_len+1:]
                            line_len=len(line_list)
                        #while 終わり（1行の末尾の処理）
                        #余りは次の行と連結
                        text=list_to_sent(line_list,0,line_len)
                        text_len=line_len
                #for終わり（ファイルの最後の行の処理）
                if text_len!=0:
                    text=preprocess_line(text)
                    f_out.write(text+'\n')
                print('total '+str(i)+' line\n')
                print_time('preprpcess end')

    return new_path


#fasttextのベクトルファイルから単語辞書とベクトル辞書の作成
def vec_to_dict(vec_path):
    print('Loading fasttext vec ...')
    s=set()
    word_indices=dict()
    indices_word=dict()
    vec_dict=dict()
    i=0
    text=''
    with open(vec_path,'r') as f:
        for line in f:
            if i!=0:
                #先頭行には単語数と次元数が書かれているので無視
                line=line.replace('\n', '').replace('\r','')
                if line[-1]==' ':
                    line=line[:-1]
                tmp_list=line.split(' ')
                word=tmp_list[0]
                str_list=tmp_list[1:]
                #辞書の作成
                #0番目はパディング用の数字なので使わないことに注意
                word_indices[word]=i
                indices_word[i]=word
                vec_dict[word]=np.array(str_list, dtype=np.float32)
            i+=1

    word_indices['#OTHER']=i
    indices_word[i]='#OTHER'
    len_words=i
    return len_words, word_indices, indices_word, vec_dict


#fasttextのベクトルを得る
#未知語の場合にはfasttextのモデル呼び出して実行
#未知語は集合に格納し，あとでファイル出力
def get_ft_vec(word, vec_dict, ft_path, bin_path):
    if word in vec_dict:
        return vec_dict[word]
    elif word in tmp_vec_dict:
        return tmp_vec_dict[word]
    else:
        KeyError_set.add(word)    #要素を追加
        cmd='echo "'+word+'" | '+ft_path+' print-word-vectors '+bin_path
        ret  =  subprocess.check_output(cmd, shell=True)
        line=ret.replace('\n', '').replace('\r','')
        if line[0]==' ':
            line=line[1:]
        if line[-1]==' ':
            line=line[:-1]
        tmp_list=line.split(' ')
        word=tmp_list[0]
        vec=tmp_list[1:]
        vec_array=np.array(vec,dtype=np.float32)
        tmp_vec_dict[word]=vec_array

        return vec_array


#単語から辞書IDを返す
def search_word_indices(word, word_to_id):
    if word in word_to_id:
        return word_to_id[word]
    else:
        return word_to_id['#OTHER']



#テストデータの前準備
def prepare_test(test_path, ch_path):

    th_len =maxlen_words/2    #テストの際の長さの閾値
    test_f_sentences = list()
    test_r_sentences = list()

    #テストデータへの読み込みと前処理
    #テストデータは学習データと異なり容量大きくないので一気に読み込んでいる
    #テストデータは1行1問で1行に<>が1つのみ
    test_file = open(test_path)
    test_data = test_file.read().lower().replace('\r','')
    test_file.close()

    ch_file= open(ch_path)
    ch_data= ch_file.read().lower().replace('\r','')
    ch_file.close()

    test_lines = test_data.split('\n')

    ch_lines = ch_data.split('\n')
    ans_list=list()
    ch_list=list()
    line_num=0

    for line in test_lines:
        if (line.count('<')*line.count('>')==1):
            mark_start=line.find('<')
            mark_end=line.find('>')
            ch_tmp_line=ch_lines[line_num]

            before=line[:mark_start]
            after=line[mark_end+1:]
            ans=line[mark_start+1:mark_end]
            choi=ch_tmp_line[ch_tmp_line.find('<')+1:ch_tmp_line.find('>')]

            before=preprocess_line(before)
            after=preprocess_line(after)
            ans=preprocess_line(ans)
            choices=choi.split(' ### ')
            flag=0
            tmp_choi=''
            for x in choices:
                x=preprocess_line(x)
                tmp_choi=tmp_choi+x+' ### '
                if x.count(' ')>0:
                    flag=-1
            if(flag==0):
                test_f_line=before.split(' ')
                test_r_line=after.split(' ')
                if (len(test_f_line)>=th_len) and (len(test_r_line)>=th_len):
                    if (len(test_f_line)>maxlen_words):
                        test_f_line=test_f_line[-1*maxlen_words:]
                    if (len(test_r_line)>maxlen_words):
                        test_r_line=test_r_line[:maxlen_words]
                    test_f_sentences.append(test_f_line)
                    test_r_sentences.append(test_r_line[::-1])
                    #テスト対象のデータの答えと選択肢をリストに格納
                    ans_list.append(ans)
                    choi=tmp_choi[:-5]  #末尾のシャープとかを削除
                    ch_list.append(choi)
                    #テスト対象となるデータのみを出力
                    with open(save_path+'testdata.txt', 'a') as data:
                        data.write(line+'\n')
        line_num+=1

    return test_f_sentences, test_r_sentences, ans_list, ch_list


#テストデータのベクトル化
def make_test_data(f_sent, r_sent, word_to_id):
    test_f_x = np.zeros((1, maxlen_words))
    test_r_x = np.zeros((1, maxlen_words))
    for t, word in enumerate(f_sent):
        tmp_index = search_word_indices(word, word_to_id)
        if(len(f_sent)<maxlen_words):
            test_f_x[0, t+maxlen_words-len(f_sent)] = tmp_index
        else:
            test_f_x[0, t] = tmp_index
    for t, word in enumerate(r_sent):
        tmp_index = search_word_indices(word, word_to_id)
        if(len(f_sent)<maxlen_words):
            test_r_x[0, t+maxlen_words-len(r_sent)] = tmp_index
        else:
            test_r_x[0, t] = tmp_index
    return test_f_x, test_r_x


#2つのベクトルのコサイン類似度を返す
def calc_similarity(pred_vec, ans_vec):
    len_p=np.linalg.norm(pred_vec)
    len_a=np.linalg.norm(ans_vec)
    if len_p==0 or len_a==0:
        return 0.0
    return np.dot(pred_vec/len_p, ans_vec/len_a)


#全単語の中からベクトルの類似度の高い順にファイル出力（あとで考察用）し，
#上位1語とその類似度，選択肢の各語の順位と類似度を返す
def print_and_get_rank(pred_vec, choices, fname, vec_dict, ft_path, bin_path, id_to_word):
    dict_all=dict()
    dict_ch=dict()
    choi_list=choices.split(' ### ')
    for i in range(len_words):
        if i!=0:
            word=id_to_word[i]
            dict_all[word]=calc_similarity(pred_vec, get_ft_vec(word, vec_dict, ft_path, bin_path))
    for x in choi_list:
        dict_ch[x]=calc_similarity(pred_vec, get_ft_vec(x, vec_dict, ft_path, bin_path))
    list_ch = sorted(dict_ch.items(), key=lambda x: x[1])
    list_all = sorted(dict_all.items(), key=lambda x: x[1])
    with open(fname, 'a') as file:
        for w,sim in list_all:
            str=w+ ' ### '
            file.write(str)
        file.write('\n')
    return (list_all[0], list_ch)
    #返り値は(単語, 類似度), {(単語, 類似度),(単語, 類似度)...}


#単語とランクリストから単語の順位をstring型で返す
def word_to_rank(word, ra_list):
    str_num=''
    if word in ra_list:
        str_num=str(ra_list.index(word))
    else:
        #無いときは-1
        str_num='-1'

    return str_num


#全単語の内類似度1語の正誤をファイル書き込み，正誤結果を返す
def calc_rank1word(top, ans, ra_list, ans_sim):
    rank_top=word_to_rank(top[0], ra_list)
    rank_ans=word_to_rank(ans, ra_list)
    out=''
    with open(save_path+'rankOK.txt', 'a') as rOK:
        with open(save_path+'rankNG.txt', 'a') as rNG:
            out='pred= ('+top[0]+', '+rank_top+', '+str(top[1])+')   '+'ans= ('+ans+', '+rank_ans+', '+str(ans_sim)+')\n'
            if top[0]==ans:
                rOK.write(out)
                OK_num=1
            else:
                rNG.write(out)
                OK_num=0
    return OK_num


#選択肢で選んだ際の正誤をファイル書き込み，正誤結果を返す
#choices=[(単語,類似度),(単語,類似度), ...]の形式
def calc_rank4choices(choices, ans, ra_list, ans_sim):
    pred=choices[0][0]
    rank_ans=word_to_rank(ans, ra_list)
    out='ans= ('+ans+', '+rank_ans+', '+str(ans_sim)+')    '+'choices='
    for word,sim in choices:
        out=out+'('+word+', '+word_to_rank(word, ra_list)+', '+str(sim)+')  '
    with open(save_path+'choicesOK.txt', 'a') as cOK:
        with open(save_path+'choicesNG.txt', 'a') as cNG:
            out=out+'\n'
            if pred==ans:
                cOK.write(out)
                OK_num=1
            else:
                cNG.write(out)
                OK_num=0
    return OK_num


#正解率の計算結果を文字列で返す
def calc_acc(rank_file, ans_list, preds_list, top_list, choice_list, vec_dict, ft_path, bin_path):
    sent_i=0
    rankOK=0
    choiOK=0

    with open(rank_file,'r') as rank:
        for line in rank:
            rank_line=line.lower().replace('\n','').replace('\r','')
            rank_list=rank_line.split(' ### ')
            ans=ans_list[sent_i]
            ans_sim=calc_similarity(preds_list[sent_i], get_ft_vec(ans, vec_dict, ft_path, bin_path))
            rankOK+=calc_rank1word(top_list[sent_i], ans, rank_list, ans_sim)
            choiOK+=calc_rank4choices(choice_list[sent_i], ans, rank_list, ans_sim)
            sent_i+=1

    rank_acc=1.0*rankOK/sent_i
    choi_acc=1.0*choiOK/sent_i

    rankNG=sent_i - rankOK
    choiNG=sent_i - choiOK

    rank_result='rank: '+str(rank_acc)+' ( OK: '+str(rankOK)+'   NG: '+str(rankNG)+' )\n'
    choi_result='choi: '+str(choi_acc)+' ( OK: '+str(choiOK)+'   NG: '+str(choiNG)+' )\n'

    result=rank_result+choi_result

    return result


#テスト
def model_test(model, test_path, ch_path, word_to_id, vec_dict, ft_path, bin_path, id_to_word):
    #テストデータの前準備
    f_sent, r_sent, ans_list, ch_list = prepare_test(test_path, ch_path)
    sent_num=len(f_sent)
    print('test_data_num=',sent_num)
    preds_list=list()
    top_list=list()
    choice_list=list()
    #テストの実行
    rank_file=save_path+'rank.txt'
    for i in range(sent_num):
        print('test_loop_i=',i)
        f_testX, r_testX = make_test_data(f_sent[i], r_sent[i], word_to_id)
        preds = min_model.predict([f_testX, r_testX], verbose=0)
        preds_list.append(preds)
        #予測結果の格納
        tmp=print_and_get_rank(preds, ch_list[i], rank_file, vec_dict, ft_path, bin_path, id_to_word)
        top=tmp[0]
        choice=tmp[1]
        top_list.append(top)
        choice_list.append(choice)
    #正解率の計算，ファイル出力
    result_str=calc_acc(rank_file, ans_list, preds_list, top_list, choice_list, vec_dict, ft_path, bin_path)

    return result_str




#----- いわゆるmain部みたいなの -----

# 0.いろいろ前準備
#開始時刻のプリント
start_time=print_time('all start')
start_time_str = start_time.strftime('%Y_%m_%d_%H%M')

#モデルとか結果とかを格納するディレクトリの作成
argvs = sys.argv
argc = len(argvs)
if argc <3:    #ファイル名 min_modelのパス wの長さで3つ必要(python は含まれない)
    print('### ERROR: invalid argument! ###')
min_model_path=argvs[1] #ここ実行時の第二引数、〜〜.jsonのファイル
#例： 2018_01_04_1450epoch100_e100_w10_add_bilstm_den1/min_model/
maxlen_words=int(argvs[2])
save_path=min_model_path[:min_model_path.find('min_model')]   #save_pathはmin_modelの手前の/まで
save_path=save_path+'TEST_'


#学習データの候補
train_big='../corpus/WikiSentWithEndMark1.txt'   # 約5.8GB，約2000万行
train_enwiki='../corpus/enwiki.txt'   # 約24GB，1行のみ，約435億単語(約237種類)
train_mid='../corpus/miniWiki_tmp8.txt'   # 約1.5MB，約5000行
train_small='../corpus/nietzsche.txt'   # 約600KB，約1万行
train_test='../corpus/mini_text8.txt'

train_text8='../corpus/text8.txt'   # 約95MB 1行のみ, 約1700万単語(約7万これ違う種類)  http://mattmahoney.net/dc/text8.zip



# 1.学習データの前処理など
tmp_path = train_text8     #使用する学習データ
print('Loading  '+tmp_path)
train_path=preprocess(tmp_path)



# 2.fasttextのロードと辞書の作成
'''
https://github.com/facebookresearch/fastText
このfastextを事前に実行しておき，その結果を利用
'''
ft_path='../../FastText/fastText-0.1.0/fasttext'

#ベクトルファイルの候補
vec_enwiki='../../FastText/Model/enwiki_dim'+str(vec_size)+'_minC0.vec'
bin_enwiki='../../FastText/Model/enwiki_dim'+str(vec_size)+'_minC0.bin'
vec_text8='../../FastText/Model/text8_dim'+str(vec_size)+'_minC0.vec'
bin_text8='../../FastText/Model/text8_dim'+str(vec_size)+'_minC0.bin'

#実際に使うもの
vec_path=vec_text8
bin_path=bin_text8

len_words, word_to_id, id_to_word, vec_dict=vec_to_dict(vec_path)

#embeddingで用いる，単語から行列への変換行列
embedding_matrix = np.zeros((len_words+1, vec_size))
for i in range(len_words):
    if i!=0:
        #IDが0の単語が存在しないので0は飛ばす
        embedding_matrix[i] = get_ft_vec(id_to_word[i], vec_dict, ft_path, bin_path)

end_data=print_time('prepare data and fasttext end')



# 5.val_loss最小モデルのロード
min_model_file=min_model_path+'my_model.json'
min_weight_file=min_model_path+'my_model.h5'
print('Loading  '+min_model_file)

json_string = open(min_model_file).read()
min_model = model_from_json(json_string)
min_model.load_weights(min_weight_file)
optimizer = RMSprop()
min_model.compile(loss='categorical_crossentropy', optimizer=optimizer)

end_load=print_time('Load min_model end')



# 6.テストの実行
test_path = '../corpus/ans_all2000_2016.txt'     #答えつきテストデータ
ch_path= '../corpus/choi_all2000_2016.txt'     #選択肢つきテストデータ

result=model_test(min_model, test_path, ch_path, word_to_id, vec_dict, ft_path, bin_path, id_to_word)
print(result)

with open(save_path+'keyerror_words.txt', 'w') as f_key:
    for word in KeyError_set:
        f_key.write(word+'\n')


end_test=print_time('test end')

#7.実行結果まとめのファイル書き込み
with open(save_path+'summary.txt', 'a') as f:
    f.write('result\n'+ result+'\n')
