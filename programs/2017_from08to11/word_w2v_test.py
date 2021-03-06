# -*- coding: utf-8 -*-

'''
word2vecでベクトルにして学習するモデル
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
from gensim.models.wrappers import FastText
import keras
import numpy as np
import re
import random
import sys
import datetime
import gensim



start_time=datetime.datetime.today()
print('all_start = ',start_time)
today_str = start_time.strftime("%Y_%m_%d_%H%M")

my_epoch=1

#today_str=today_str+'_epoch'+str(my_epoch)

#読み込むもの

train_big='./WikiSentWithEndMark1.txt'   # 約5.8GB，約2000万行
train_mid='./miniWiki_tmp8.txt'   # 約1.5MB，約5000行
train_small='./nietzsche.txt'   # 約600KB，約1万行



train_path = train_mid        #学習データ
test_path = './tmp_testdata_after.txt'     #答えつきテストデータ
ch_path= './tmp_choices_after.txt'     #選択肢つきテストデータ



#出力されるもの

dic_path='../wordlist_WikiSentWithEndMark1.txt'    #辞書データ？単語リスト的な


'''


modelのデータ
    日付_model.json  #モデルの構造とか
    日付_model.h5    #学習後の重みとか


実際にテストに使用したデータ
    日付_testdata.txt


予測結果
    日付_rank.txt     #全単語を確率順に並べたもの
    日付_preds.txt    #サンプリング後，予測した単語1語//これいらない
    日付_sampOK.txt    #正解：1語サンプリング
    日付_sampNG.txt    #不正解：1語サンプリング
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


tmp_path=train_path[:-4]+'_cleaned.txt'


#学習データへの前処理
print('Preprpcessing training data...')
with open(train_path) as f_in:
    with open(tmp_path, 'w') as f_out:
        for line in f_in:
            text=line.lower()
            text = re.sub(r"[^a-z ]", "", text)
            text = re.sub(r"[ ]+", " ", text)
            f_out.write(text)

train_path=tmp_path

#word2vecの学習
vec_size=100

print('Learning word2vec...')
sentences = gensim.models.word2vec.Text8Corpus(train_path)
'''
このText8Corpusは半角スペースなどで区切られたファイルを引数にとる
単語で分割したリストを返す．
詳しくは https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/word2vec.py
'''

w2v_model = gensim.models.word2vec.Word2Vec(sentences, size=vec_size, window=5, workers=4, min_count=0)
w2v_model.save(today_str+'_w2v.model')




'''






https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/wrappers/fasttext.py




FastTextの方は要調査
ft_model = FastText.train('/Users/kofola/fastText/fasttext', corpus_file='text8')

`ft_path` is the path to the FastText executable, e.g. `/home/kofola/fastText/fasttext`.
`corpus_file` is the filename of the text file to be used for training the FastText model.
Expects file to contain utf-8 encoded text.

ft_pathはFastTextへの実行パス？どういうこと？
 → https://groups.google.com/forum/#!topic/gensim/XX02Hzb_2l0
    ここのサイト参考で解決しそう．ft_pathはfasttextのファイルへのパスのこと
    gensimではなくまず本家(facebook)のgit cloneしたディレクトリでやるとよさげ

http://gensim.narkive.com/dp1Mhp6z/gensim-9070-fasttext-wrapper-in-gensim

fasttext本家のreadmeは一度読んどくべきかも

corpus_fileは学習データみたい

FastTextはcbowとskipgramの二つの学習方法があるがデフォルトではcbow

ft_model.save(today_str+'_ft.model')

FastTextはgensimのpythonのやつとfacebookのC++のやつとある
さらにはsalestockってところのやつもある？書き方がバラバラなので検索注意





ここらへんいつか使うかも
https://qiita.com/iss-f/items/aec567ee5c79464413dc
https://stackoverflow.com/questions/32759712/how-to-find-the-closest-word-to-a-vector-using-word2vec
'''

print_time('word2vec end')



#辞書の作成（大きなコーパスに対応するようにしてる）
print('Making dic...')
s=set()
with open(train_path) as f:
    for line in f:
        text = text.replace("\n", " ").replace('\r','')
        text = re.sub(r"[ ]+", " ", text)
        text_list=text.split(" ")
        tmp_set=set(text_list)
        s.update(tmp_set)    #和集合をとる

words = sorted(list(s))

with open(dic_path, 'a') as f_d:
    tmp_dic_i=0
    for x in words:
        tmp_dic_i+=1
        f_d.write(str(x)+' ')
        if tmp_dic_i % 100 ==0:
            f_d.write('\n')
    f_d.write('\n')

words.append("#OTHER")
len_words=len(words)
print('kind of words:', len_words)
word_indices = dict((c, i) for i, c in enumerate(words))
indices_word = dict((i, c) for i, c in enumerate(words))



print_time('make dic end')

#単語から辞書IDを返す
def search_word_indices(word):
    if word in word_indices:
        return word_indices[word]
    else:
        return word_indices["#OTHER"]



KeyError_set=set()


# モデルの構築
maxlen_words = 10
print('Build model...')
f_input=Input(shape=(maxlen_words, vec_size))
f_layer=LSTM(128,)(f_input)

r_input=Input(shape=(maxlen_words, vec_size))
r_layer=LSTM(128,)(r_input)

merged_layer=add([f_layer, r_layer])

out_layer=Dense(len_words,activation='softmax')(merged_layer)

my_model=Model([f_input, r_input], out_layer)

optimizer = RMSprop(lr=0.01)
my_model.compile(loss='categorical_crossentropy', optimizer=optimizer)


#word2vecのベクトルを得る
#未知語の場合には[0,0,0, ... ,0]みたいなやつにとりあえずしてる
#未知語は集合に格納し，あとでファイル出力
#要改良
def get_w2v_vec(word):
    if word in w2v_model.wv.vocab:
        return w2v_model.wv[word]
    else:
        KeyError_set.add(word)    #要素を追加
        return np.zeros((vec_size),dtype=np.float32)


#学習データをベクトル化してモデルを学習
def model_fit(midtext, model):
    midtext = re.sub(r"[ ]+", " ", midtext)
    if(midtext[0]==' '):
        midtext=midtext[1:]
    text_list=midtext.split(" ")
    f_sentences = []
    r_sentences = []
    next_words = []
    step = 3
    len_text=len(text_list)
    if (len_text - maxlen_words*2 -1) > 0:
        for i in range(0, len_text - maxlen_words*2 -1, step):
            f_sentences.append(text_list[i: i + maxlen_words])
            r_sentences.append(text_list[i + maxlen_words+1: i + maxlen_words+1+maxlen_words][::-1]) #逆順のリスト
            next_words.append(text_list[i + maxlen_words])
        len_sent=len(f_sentences)
        
        f_X = np.zeros((len_sent, maxlen_words, vec_size),dtype=np.float32) #word2vecはfloat32らしい
        r_X = np.zeros((len_sent, maxlen_words, vec_size),dtype=np.float32)
        y = np.zeros((len_sent, len_words), dtype=np.bool)
        for i, sentence in enumerate(f_sentences):
            for t, word in enumerate(sentence):
                f_X[i, t] = get_w2v_vec(word)
            y[i, search_word_indices(next_words[i])] = 1
        
        for i, sentence in enumerate(r_sentences):
            for t, word in enumerate(sentence):
                r_X[i, t] = get_w2v_vec(word)
        # モデルの学習
        model.fit([f_X,r_X], y, batch_size=128, epochs=1)





# 学習（大きなコーパスに対応するようにしてる）
for ep_i in range(my_epoch):
    print('\nEPOCH='+str(my_epoch)+'\n')
    with open(train_path) as f:
        read_i=0
        text=""
        for line in f:
            read_i+=1
            t_line = line.replace("\n", " ").replace('\r','')
            t_line = re.sub(r"[ ]+", " ", t_line)
            text=text+' '+t_line
            # 1000行ごとに学習
            if(read_i % 1000==0):
                model_fit(text, my_model)
                text=""
                
    #最後の余りを学習
    if(len(text)>0):
        model_fit(text, my_model)
    text=""

print_time('fit end')



#モデルの保存
model_json_str = my_model.to_json()
file_model=today_str+'_my_model'
open(file_model+'.json', 'w').write(model_json_str)
my_model.save_weights(file_model+'.h5')


#サンプリング関数
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


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
            with open(today_str+'_testdata.txt', 'a') as data:
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
            str=indices_word[k]+ " ### "
            file.write(str)
        file.write('\n')


#テストデータの作成（ベクトル化）
#テストの実行
print('Test starts ...')

preds_list=[]

for i in range(sents_num):
    test_f_x = np.zeros((1, maxlen_words,vec_size),dtype=np.float32)
    test_r_x = np.zeros((1, maxlen_words,vec_size),dtype=np.float32)
    for t, word in enumerate(test_f_sentences[i]):
        tmp_vec = get_w2v_vec(word)
        if(len(test_f_sentences[i])<maxlen_words):
            test_f_x[0, t+maxlen_words-len(test_f_sentences[i])] = tmp_vec
        else:
            test_f_x[0, t] = tmp_vec
    for t, word in enumerate(test_r_sentences[i]):
        tmp_vec = get_w2v_vec(word)
        if(len(test_f_sentences[i])<maxlen_words):
            test_r_x[0, t+maxlen_words-len(test_r_sentences[i])] = tmp_vec
        else:
            test_r_x[0, t] = tmp_vec
    preds = my_model.predict([test_f_x,test_r_x], verbose=0)[0]
        
    print_rank(preds, today_str+'_rank.txt')
    test_next_index = sample(preds)
    test_next_word = indices_word[test_next_index]
    
    #サンプリングでの予測語をリストに格納
    preds_list.append(test_next_word)

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



#サンプリングで選んだ1語の正誤をファイル書き込み，正誤結果を返す
def calc_samp1word(pred, ans, list_rank):
    rank_pred=word_to_rank(pred, list_rank)
    rank_ans=word_to_rank(ans, list_rank)
    out=''
    with open(today_str+'_sampOK.txt', 'a') as sOK:
        with open(today_str+'_sampNG.txt', 'a') as sNG:
            out='pred= '+pred+' : '+rank_pred+'     '+'ans= '+ans+' : '+rank_ans+'\n'
            if pred==ans:
                sOK.write(out)
                OK_num=1
            else:
                sNG.write(out)
                OK_num=0
    return OK_num

#確率で選んだ1語の正誤をファイル書き込み，正誤結果を返す
def calc_rank1word(pred, ans, list_rank):
    rank_pred=word_to_rank(pred, list_rank)
    rank_ans=word_to_rank(ans, list_rank)
    out=''
    with open(today_str+'_rankOK.txt', 'a') as rOK:
        with open(today_str+'_rankNG.txt', 'a') as rNG:
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

    with open(today_str+'_choicesOK.txt', 'a') as cOK:
        with open(today_str+'_choicesNG.txt', 'a') as cNG:
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
sampOK=0
rankOK=0
choiOK=0

with open(today_str+'_rank.txt',"r") as rank:
    for line in rank:
        rank_line=line.lower().replace('\n','').replace('\r','')
        rank_list=rank_line.split(' ### ')
        sampOK+=calc_samp1word(preds_list[sent_i], ans_list[sent_i], rank_list)
        rankOK+=calc_rank1word(preds_list[sent_i], ans_list[sent_i], rank_list)
        choiOK+=calc_rank4choices(ch_list[sent_i], ans_list[sent_i], rank_list)
        sent_i+=1

samp_acc=1.0*sampOK/sent_i
rank_acc=1.0*rankOK/sent_i
choi_acc=1.0*choiOK/sent_i

sampNG=sent_i - sampOK
rankNG=sent_i - rankOK
choiNG=sent_i - choiOK

samp_result='samp: '+str(samp_acc)+' ( OK: '+str(sampOK)+'   NG: '+str(sampNG)+' )\n'
rank_result='rank: '+str(rank_acc)+' ( OK: '+str(rankOK)+'   NG: '+str(rankNG)+' )\n'
choi_result='choi: '+str(choi_acc)+' ( OK: '+str(choiOK)+'   NG: '+str(choiNG)+' )\n'


result=samp_result+rank_result+choi_result

end_time=datetime.datetime.today()
diff_time=end_time-start_time

with open(today_str+'_result.txt', 'a') as rslt:
    rslt.write(result+'\ntotal time ='+str(diff_time))

print(result)


#未知語と認識された語を出力
with open(today_str+'_KeyErrorWords.txt', 'a') as f_key:
    for x in KeyError_set:
        f_key.write(x+'\n')


print('all_end = ',end_time)

plot_model(my_model, to_file=today_str+'model.png', show_shapes=True)

print('total =',diff_time)
