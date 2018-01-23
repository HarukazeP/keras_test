# -*- coding: utf-8 -*-

'''
学習途中で中断したモデルをロードして続きから学習するやつ

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
from keras.layers import LSTM
from keras.layers import add, concatenate
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras.utils.vis_utils import plot_model
import keras
import numpy as np
import re
import sys
import datetime
import os
import os.path
import matplotlib
matplotlib.use('Agg')    #これをpyplotより先に書くことでサーバでも動くようにしている
import matplotlib.pyplot as plt
import subprocess

#----- グローバル変数一覧 -----
my_epoch=100
vec_size=100
maxlen_words = 5
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


#1行の文字列を学習データの形式に変換
def make_train_data(line, len_words, word_to_id, vec_dict, ft_path, bin_path):
    line=line.replace('\n','').replace('\r','')
    if line[0]==' ':
        line=line[1:]
    if line[-1]==' ':
        line=line[:-1]
    text_list=line.split(' ')
    f_sentences = list()
    r_sentences = list()
    next_words = list()
    step=3
    len_text=len(text_list)
    if (len_text - maxlen_words*2 -1) > 0:
        for i in range(0, len_text - maxlen_words*2 -1, step):
            f=text_list[i: i + maxlen_words]
            r=text_list[i + maxlen_words+1: i + maxlen_words+1+maxlen_words]
            n=text_list[i + maxlen_words]
            f_sentences.append(f)
            r_sentences.append(r[::-1]) #逆順のリスト
            next_words.append(n)
        len_sent=len(f_sentences)

        f_X = np.zeros((len_sent, maxlen_words), dtype=np.int)
        r_X = np.zeros((len_sent, maxlen_words), dtype=np.int)
        Y = np.zeros((len_sent, vec_size), dtype=np.float32)
        for i, sentence in enumerate(f_sentences):
            Y[i] = get_ft_vec(next_words[i], vec_dict, ft_path, bin_path)
            for t, word in enumerate(sentence):
                f_X[i, t] = search_word_indices(word, word_to_id)


        for i, sentence in enumerate(r_sentences):
            for t, word in enumerate(sentence):
                r_X[i, t] = search_word_indices(word, word_to_id)

    return f_X, r_X, Y


#loss, val_lossの追加更新
def conect_hist(list_loss, list_val_loss, new_history):
    list_loss.extend(new_history.history['loss'])
    list_val_loss.extend(new_history.history['val_loss'])


#1行10000単語までのファイルから1行ずつ1回学習する
#lossやval_lossは各行の学習結果の中央値を返す
def model_fit_once(train_path, my_model, len_words, word_to_id, vec_dict, ft_path, bin_path):
    tmp_loss_list=list()
    tmp_val_loss_list=list()

    with open(train_path) as f:
        for line in f:
            line = re.sub(r'[ ]+', ' ', line)
            if line.count(' ')>maxlen_words*10:
                f_trainX, r_trainX, trainY = make_train_data(line, len_words, word_to_id, vec_dict, ft_path, bin_path)
                tmp_hist=my_model.fit([f_trainX,r_trainX], trainY, batch_size=128, epochs=1, validation_split=0.1)
                conect_hist(tmp_loss_list, tmp_val_loss_list, tmp_hist)

    loss=np.median(np.array(tmp_loss_list, dtype=np.float32))
    val_loss=np.median(np.array(tmp_val_loss_list, dtype=np.float32))
    with open(save_path+'loss.txt', 'a') as f_loss:
            f_loss.write('loss='+str(loss)+'  , val_loss='+str(val_loss)+'\n')
    return loss, val_loss


#my_epochの数だけ学習をくりかえす
def model_fit_loop_continue(now_epoch, train_path, my_model, len_words, word_to_id, vec_dict, ft_path, bin_path):
    list_loss=list()
    list_val_loss=list()
    for ep_i in range(now_epoch,my_epoch):
        print('\nEPOCH='+str(ep_i+1)+'/'+str(my_epoch)+'\n')
        loss, val_loss=model_fit_once(train_path, my_model,len_words, word_to_id, vec_dict, ft_path, bin_path)
        list_loss.append(loss)
        list_val_loss.append(val_loss)

        #モデルの保存
        dir_name=save_path+'Model_'+str(ep_i+1)
        os.mkdir(dir_name)

        model_json_str = my_model.to_json()
        file_model=dir_name+'/my_model'
        open(file_model+'.json', 'w').write(model_json_str)
        my_model.save_weights(file_model+'.h5')


    return list_loss, list_val_loss


# 損失の履歴をプロット
def plot_loss(list_loss, list_val_loss, title='model loss'):
    plt.plot(list_loss, color='blue', marker='o', label='loss')
    plt.plot(list_val_loss, color='green', marker='o', label='val_loss')
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(save_path+'loss_graph.png')
    #plt.show()は行うとプログラム中断されるからNG


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







#----- いわゆるmain部みたいなの -----

# 0.いろいろ前準備
#開始時刻のプリント
start_time=print_time('all start')
start_time_str = start_time.strftime('%Y_%m_%d_%H%M')

#モデルとか結果とかを格納するディレクトリの作成
argvs = sys.argv
argc = len(argvs)
if argc <3:    #ファイル名 now_modelのパス wの長さで3つ必要(python は含まれない)
    print('### ERROR: invalid argument! ###')
now_model_path=argvs[1] #ここ実行時の第二引数、〜〜.jsonのファイル
#例： 2018_01_04_1450epoch100_e100_w10_add_bilstm_den1/Model_20/
maxlen_words=int(argvs[2])
save_path=now_model_path[:now_model_path.find('/Model_')]   #save_pathは/の手前まで
save_path=save_path+'NEXT_'


#学習データの候補
train_big='../corpus/WikiSentWithEndMark1.txt'   # 約5.8GB，約2000万行
train_enwiki='../corpus/enwiki.txt'   # 約24GB，1行のみ，約435億単語(約237種類)
train_mid='../corpus/miniWiki_tmp8.txt'   # 約1.5MB，約5000行
train_small='../corpus/nietzsche.txt'   # 約600KB，約1万行
train_test='../corpus/mini_text8.txt'

train_text8='../corpus/text8.txt'   # 約95MB 1行のみ, 約1700万単語(約7万種類)  http://mattmahoney.net/dc/text8.zip



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



# 3.モデルのロード
now_model_file=now_model_path+'my_model.json'
now_weight_file=now_model_path+'my_model.h5'
print('Loading  '+now_model_file)

json_string = open(now_model_file).read()
now_model = model_from_json(json_string)
now_model.load_weights(now_weight_file)
optimizer = RMSprop()
now_model.compile(loss='mean_squared_error', optimizer=optimizer)



# 4.モデルの学習再開
st_train=print_time('train start')
now_epoch_str=now_model_path[now_model_path.find('/Model_'):]
now_epoch_str = re.sub(r'[^0-9]', '', now_epoch_str)
now_epoch=int(now_epoch_str)-1
loss, val_loss=model_fit_loop_continue(now_epoch, train_path, now_model, len_words, word_to_id, vec_dict, ft_path, bin_path)
plot_loss(loss, val_loss)

end_train=print_time('train end')






#7.実行結果まとめのファイル書き込み
#下記内容をファイルにまとめて出力
'''
・実行したプログラム名
・実施日時（開始時刻）
・読み込んだ学習データ
・単語数
・全学習回数
・val_loss最小の学習回数

・テスト結果

・modelの概要

・学習データの前処理，辞書の作成ににかかった時間
・fasttextのロードとembedding_matrixの作成にかかった時間
・学習にかかった時間（ベクトル化も含む）
・val_loss最小モデルのロードにかかった時間
・テストにかかった時間（ベクトル化，正解率とかも含む）
・全合計かかった時間
'''



with open(save_path+'summary.txt', 'a') as f:
    f.write('Result of '+program_name+'\n\n')

    f.write('start_time = '+ start_time_str+'\n')
    f.write('epoch = '+str(my_epoch)+'\n')
    f.write('train_data = '+ train_path+'\n')
    f.write('kind of words ='+str(len_words)+'\n')

    f.write('TIME train = '+ str(end_train-st_train)+'\n')


    end_time=print_time('all end')
    f.write('TIME total = '+ str(end_time-start_time)+'\n')
