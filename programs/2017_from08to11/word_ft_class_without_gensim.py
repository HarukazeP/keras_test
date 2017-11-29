# -*- coding: utf-8 -*-

'''
fasttextでベクトルにして学習するモデル
embeddingレイヤーでfasttextのベクトルを利用
'''

from __future__ import print_function
from keras.models import Sequential, Model
from keras.models import model_from_json
from keras.layers import Dense, Activation, Input, Embedding
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
import os

my_epoch=2

vec_size=100

start_time=datetime.datetime.today()
print('all_start = ',start_time)
today_str = start_time.strftime("%Y_%m_%d_%H%M")

today_str=today_str+'epoch'+str(my_epoch)
os.mkdir(today_str)
today_str='./'+today_str+'/'
#日付名+epoch数のフォルダを作成し、結果はすべてそこに格納

#読み込むもの
'''
train_big='../corpus/WikiSentWithEndMark1.txt'   # 約5.8GB，約2000万行
train_mid='../corpus/miniWiki_tmp8.txt'   # 約1.5MB，約5000行
train_small='../corpus/nietzsche.txt'   # 約600KB，約1万行

train_text8='../corpus/text8.txt'   # 約95MB 1行のみ　http://mattmahoney.net/dc/text8.zip
'''


train_text8_cleaned='../corpus/text8_cleaned.txt'

train_path = train_text8_cleaned        #学習データ
vec_path='/home/tamaki/M1/Keras/mine2017_8to11/2017_11_28_2032epoch100/vec.txt'    #fasttextのベクトルを書き出したファイル

test_path = '../corpus/tmp_testdata_after.txt'     #答えつきテストデータ
ch_path= '../corpus/tmp_choices_after.txt'     #選択肢つきテストデータ

print('Loading  '+train_path)




'''
#出力されるもの

modelのデータ
    日付_model.json  #モデルの構造とか
    日付_model.h5    #学習後の重みとか


実際にテストに使用したデータ
    日付_testdata.txt


予測結果
    日付_rank.txt     #全単語を確率順に並べたもの
    日付_preds.txt    #サンプリング後，予測した単語1語//これいらない
    日付_rankOK.txt    #正解：1語確率
    日付_rankNG.txt    #不正解：1語確率
    日付_choicesOK.txt    #正解：4択確率
    日付_choicesNG.txt    #不正解：4択確率
    日付_result.txt   #全部の正解率とかを記録したファイル

'''

#時間表示
def print_time(str1):
    today=datetime.datetime.today()
    print(str1)
    print(today)



def conect_hist(list_loss, list_val_loss, new_history):
    list_loss.extend(new_history.history['loss'])
    list_val_loss.extend(new_history.history['val_loss'])


def str_to_numpy(str_numpy):
    str_numpy=str_numpy.replace(' [ ', '').replace(' ] ', '')
    str_numpy=str_numpy.replace('[ ', '').replace('] ', '')
    str_numpy=str_numpy.replace(' [', '').replace(' ]', '')
    str_numpy=str_numpy.replace('[', '').replace(']', '')
    str_list=str_numpy.split(' ')
    len_list=len(str_list)
    #print (str_list)
    array=np.zeros(len_list,dtype=np.float32)
    for i in range(len_list):
        array[i]=float(str_list[i])

    return array

    '''
    #ValueError: could not convert string to float:
    str_array=np.array(str_list)
    return str_array.astype(np.float32)
    '''

#fasttextベクトルの用意と辞書の作成
s=set()
word_indices=dict()
indices_word=dict()
vec_dict=dict()
i=0
text=""
with open(vec_path,"r") as f:
    for line in f:
        text=text+line
        if line.find(']')>=0:
            text=text.replace("\n", '').replace('\r','')
            text=re.sub(r"[ ]+", " ", text)
            tmp_list=text.split(" > ")
            word=tmp_list[0]
            str_numpy=tmp_list[1]
            #print(str_numpy)
            #辞書の作成
            #0番目はパディング用の数字なので使わないことに注意
            if word!='':
                word_indices[word]=i+1
                indices_word[i+1]=word
                #print ('word_i=',i)
                vec_dict[word]=str_to_numpy(str_numpy)
                text=""
                i+=1

word_indices['#OTHER']=i+1
indices_word[i+1]='#OTHER'
len_words=i




#fasttextのベクトルを得る
#未知語の場合には[0,0,0, ... ,0]みたいなやつにとりあえずしてる
#未知語は集合に格納し，あとでファイル出力
#要改良
KeyError_set=set()
def get_ft_vec(word):
    if word in vec_dict:
        return vec_dict[word]
    else:
        KeyError_set.add(word)    #要素を追加
        return np.zeros(vec_size,dtype=np.float32)


embedding_matrix = np.zeros((len_words+1, vec_size))
for i in range(len_words):
    if i!=0:
        embedding_matrix[i] = get_ft_vec(indices_word[i])
        #IDが0の単語が存在しないので0は飛ばす





print_time('make dic end')


#単語から辞書IDを返す
def search_word_indices(word):
    if word in word_indices:
        return word_indices[word]
    else:
        return word_indices["#OTHER"]




maxlen_words = 10
step = 3



# モデルの構築
print('Build model...')
f_input=Input(shape=(maxlen_words,))
f_emb=Embedding(output_dim=vec_size, input_dim=len_words+1, input_length=maxlen_words, mask_zero=True, weights=[embedding_matrix], trainable=False)(f_input)

f_layer=LSTM(128)(f_emb)

r_input=Input(shape=(maxlen_words,))
r_emb=Embedding(output_dim=vec_size, input_dim=len_words+1, input_length=maxlen_words, mask_zero=True, weights=[embedding_matrix], trainable=False)(r_input)
r_layer=LSTM(128)(r_emb)

merged_layer=add([f_layer, r_layer])

out_layer=Dense(len_words,activation='softmax')(merged_layer)

my_model=Model([f_input, r_input], out_layer)

optimizer = RMSprop()
my_model.compile(loss='categorical_crossentropy', optimizer=optimizer)


#小さいデータをベクトル化してモデルを学習
def model_fit(midtext, model):
    midtext = re.sub(r"[ ]+", " ", midtext)
    if(midtext[0]==' '):
        midtext=midtext[1:]
    text_list=midtext.split(" ")
    f_sentences = []
    r_sentences = []
    next_words = []
    len_text=len(text_list)
    if (len_text - maxlen_words*2 -1) > 0:
        for i in range(0, len_text - maxlen_words*2 -1, step):
            f_sentences.append(text_list[i: i + maxlen_words])
            r_sentences.append(text_list[i + maxlen_words+1: i + maxlen_words+1+maxlen_words][::-1]) #逆順のリスト
            next_words.append(text_list[i + maxlen_words])
        len_sent=len(f_sentences)

        f_X = np.zeros((len_sent, maxlen_words), dtype=np.int)
        r_X = np.zeros((len_sent, maxlen_words), dtype=np.int)
        y = np.zeros((len_sent, len_words), dtype=np.bool)
        for i, sentence in enumerate(f_sentences):
            for t, word in enumerate(sentence):
                f_X[i, t] = word_indices[word]
            y[i, word_indices[next_words[i]]] = 1

        for i, sentence in enumerate(r_sentences):
            for t, word in enumerate(sentence):
                r_X[i, t] = word_indices[word]
        # モデルの学習
        history=model.fit([f_X,r_X], y, batch_size=128, epochs=1, validation_split=0.1)

        return history


list_loss=list()
list_val_loss=list()
min_i=0
# 学習（大きなコーパスに対応するようにしてる）
#kerasのepochじゃなくてこのforループ回すことで先頭の学習データ2回目みたいなことしたい
#これ意味あるかどうかはわからない
for ep_i in range(my_epoch):
    print('\nEPOCH='+str(ep_i+1)+'/'+str(my_epoch)+'\n')
    with open(train_path,"r") as f:
        read_i=0
        text=""
        for line in f:
            read_i+=1
            t_line = line.lower()
            t_line = t_line.replace("\n", " ").replace('\r','')
            t_line = re.sub(r"[ ]+", " ", t_line)
            text=text+' '+t_line
            # 1000行ごとに学習
            if(read_i % 1000==0):
                my_hist=model_fit(text, my_model)
                conect_hist(list_loss, list_val_loss, my_hist)
                text=""

    #最後の余りを学習
    if(len(text)>0):
        my_hist=model_fit(text, my_model)
        conect_hist(list_loss, list_val_loss, my_hist)
    text=""

    #val_lossの最小値を記録
    new_val_loss=my_hist.history['val_loss']
    if( (ep_i!=0)and(min>new_val_loss) ):
        min=new_val_loss
        min_i=ep_i

    #モデルの保存
    dir_name=today_str+'Model_'+str(ep_i+1)
    os.mkdir(dir_name)

    model_json_str = my_model.to_json()
    file_model=dir_name+'/my_model'
    open(file_model+'.json', 'w').write(model_json_str)
    my_model.save_weights(file_model+'.h5')

print_time('fit end')



#val_lossが最小となるモデルを採用
min_model_file=today_str+'Model_'+str(min_i+1)+'/my_model.json'
min_weight_file=today_str+'Model_'+str(min_i+1)+'/my_model.h5'
json_string = open(min_model_file).read()
min_model = model_from_json(json_string)
min_model.load_weights(min_weight_file)
optimizer = RMSprop()
min_model.compile(loss='categorical_crossentropy', optimizer=optimizer)

th_len =maxlen_words/2    #テストの際の長さの閾値
test_f_sentences = []
test_r_sentences = []
sents_num=0


#テストデータへの読み込みと前処理
#テストデータは1行1問で1行に<>が1つのみ
test_file = open(test_path)
test_data = test_file.read().lower().replace('\r','')
test_file.close()

ch_file= open(ch_path)
ch_data= ch_file.read().lower().replace('\r','')
ch_file.close()

all_lines = test_data.split("\n")

ch_lines = ch_data.split("\n")
ans_list=[]
ch_list=[]
line_num=0

for line in all_lines:
    tmp=re.split('<.+>', line)
    if(len(tmp)>1):
        test_f_tmp=re.sub(r"[^a-z ]", "", tmp[0])
        test_f_tmp = re.sub(r"[ ]+", " ", test_f_tmp)
        test_r_tmp=re.sub(r"[^a-z ]", "", tmp[1])
        test_r_tmp = re.sub(r"[ ]+", " ", test_r_tmp)
        test_f_line=test_f_tmp.split(" ")
        test_r_line=test_r_tmp.split(" ")
        if (len(test_f_line)>=th_len) and (len(test_r_line)>=th_len):
            if (len(test_f_line)>maxlen_words):
                test_f_line=test_f_line[-1*maxlen_words:]
            if (len(test_r_line)>maxlen_words):
                test_r_line=test_r_line[:maxlen_words]
            test_f_sentences.append(test_f_line)
            test_r_sentences.append(test_r_line[::-1])
            sents_num+=1
            #テスト対象のデータの答えと選択肢をリストに格納
            #tmp_ans=ans_lines[line_num]
            tmp_ans=all_lines[line_num]
            tmp_ans=tmp_ans[tmp_ans.find('<')+1:tmp_ans.find('>')]
            ans_list.append(tmp_ans)
            tmp_ch=ch_lines[line_num]
            tmp_ch=tmp_ch[tmp_ch.find('<')+1:tmp_ch.find('>')]
            ch_list.append(tmp_ch)
            #テスト対象となるデータのみを出力
            with open(today_str+'testdata.txt', 'a') as data:
                data.write(line+'\n')

    line_num+=1


print('test_sentences:', sents_num)



#与えられた確率付き単語リストからランキング順に単語のみファイルへ書き込み
def print_rank(list1, fname):
    dict_A = dict((i,c) for i,c in enumerate(list1))
    list_B = sorted(dict_A.items(), key=lambda x: x[1], reverse=True)
    with open(fname, "a") as file:
        #print('Write rank ...')
        for k,v in list_B:
            if k!=0:
                #idが0は存在しないので飛ばす
                str=indices_word[k]+ " ### "
                file.write(str)
        file.write('\n')


#テストデータの作成（ベクトル化）
#テストの実行
print('Test starts ...')

preds_list=[]

for i in range(sents_num):
    test_f_x = np.zeros((1, maxlen_words))
    test_r_x = np.zeros((1, maxlen_words))
    for t, word in enumerate(test_f_sentences[i]):
        tmp_index = search_word_indices(word)
        if(len(test_f_sentences[i])<maxlen_words):
            test_f_x[0, t+maxlen_words-len(test_f_sentences[i])] = tmp_index
        else:
            test_f_x[0, t] = tmp_index
    for t, word in enumerate(test_r_sentences[i]):
        tmp_index = search_word_indices(word)
        if(len(test_f_sentences[i])<maxlen_words):
            test_r_x[0, t+maxlen_words-len(test_r_sentences[i])] = tmp_index
        else:
            test_r_x[0, t] = tmp_index
    preds = min_model.predict([test_f_x,test_r_x], verbose=0)[0]

    print_rank(preds, today_str+'rank.txt')

print_time('test end')




#単語とランクリストから単語の順位をstring型で返す
def word_to_rank(word, ra_list):
    str_num=''
    if word in ra_list:
        str_num=str(ra_list.index(word))
    else:
        #無いときは-1
        str_num='-1'

    return str_num



#ランクリストと選択肢リストから，選択肢をランクリストの何番目に現れるか（順位）つきで並べた文字列を返す
#選択肢の語がランクリストにないときは-1
def search_rank(ra_list, ch_list):
    str_rank=''
    str_num=''
    k=0
    for x in ch_list:
        str_num=word_to_rank(x, ra_list)
        str_rank=str_rank+x+': '+str_num+' ### '
    #末尾のシャープとか消す
    k=len(str_rank)-5
    str_rank=str_rank[:k]

    return str_rank



#順位付き文字列と選択肢リストから最も順位の高い単語を返す
#どの語もランクリストにないときは#OTHERを返す
def serch_highest(str_rank, ch_list):
    tmp_list=str_rank.split(' ### ')
    num_list=[]
    flag=0
    word=''
    for x in tmp_list:
        num=int(x[x.index(': ')+2:])
        num_list.append(num)
    min=max(num_list)+10    #この10に特に意味はない．単に大きい数字にしたいだけ
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


#確率で選んだ1語の正誤をファイル書き込み，正誤結果を返す
def calc_rank1word(pred, ans, list_rank):
    rank_pred=word_to_rank(pred, list_rank)
    rank_ans=word_to_rank(ans, list_rank)
    out=''
    with open(today_str+'rankOK.txt', 'a') as rOK:
        with open(today_str+'rankNG.txt', 'a') as rNG:
            out='pred= '+pred+' : '+rank_pred+'     '+'ans= '+ans+' : '+rank_ans+'\n'
            if pred==ans:
                rOK.write(out)
                OK_num=1
            else:
                rNG.write(out)
                OK_num=0
    return OK_num

#確率で選択肢から選んだ際の正誤をファイル書き込み，正誤結果を返す
def calc_rank4choices(choices, ans, list_rank):
    #まず選択肢をリストへ
    choi_list=choices.split(' ### ')
    out=search_rank(list_rank, choi_list)
    pred=serch_highest(out, choi_list)

    with open(today_str+'choicesOK.txt', 'a') as cOK:
        with open(today_str+'choicesNG.txt', 'a') as cNG:
            out=out+'\n'
            if pred==ans:
                cOK.write(out)
                OK_num=1
            else:
                cNG.write(out)
                OK_num=0
    return OK_num


#ここから正解率の計算とか
sent_i=0
rankOK=0
choiOK=0

with open(today_str+'rank.txt',"r") as rank:
    for line in rank:
        rank_line=line.lower().replace('\n','').replace('\r','')
        rank_list=rank_line.split(' ### ')
        rankOK+=calc_rank1word(preds_list[sent_i], ans_list[sent_i], rank_list)
        choiOK+=calc_rank4choices(ch_list[sent_i], ans_list[sent_i], rank_list)
        sent_i+=1

rank_acc=1.0*rankOK/sent_i
choi_acc=1.0*choiOK/sent_i

rankNG=sent_i - rankOK
choiNG=sent_i - choiOK

rank_result='rank: '+str(rank_acc)+' ( OK: '+str(rankOK)+'   NG: '+str(rankNG)+' )\n'
choi_result='choi: '+str(choi_acc)+' ( OK: '+str(choiOK)+'   NG: '+str(choiNG)+' )\n'


result=rank_result+choi_result+'\n\nmodel='+min_model_file

end_time=datetime.datetime.today()
diff_time=end_time-start_time

with open(today_str+'result.txt', 'w') as rslt:
    rslt.write(result+'\ntotal time ='+str(diff_time))

print(result)

with open(today_str+'val_loss.txt', 'w') as f1:
    f1.write(str(my_hist.history['val_loss']))

with open(today_str+'loss.txt', 'w') as f2:
    f2.write(str(my_hist.history['loss']))


print('all_end = ',end_time)
#plot_loss(list_loss, list_val_loss)
#plot_model(min_model, to_file=today_str+'model.png', show_shapes=True)



print('total =',diff_time)
