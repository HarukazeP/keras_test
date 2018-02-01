# -*- coding: utf-8 -*-

'''
fasttext�Ńx�N�g���ɂ��Ċw�K���郂�f��
embedding���C���[��fasttext�̃x�N�g���𗘗p
gensim���C�u�����͎g��Ȃ�
�P��̃x�N�g��������\�������A���f��

python    : 2.7.12
keras     : 2.0.4
gensim    : 3.0.1
tensorflow: 1.1.0

�v���O�����S�̂̍\��
    �E�O���[�o���ϐ��ꗗ
    �E�֐��Q
    �E������main���݂����Ȃ�

�v���O�����S�̗̂���
    0.���낢��O����
    1.�w�K�f�[�^�̑O����
    2.fasttext�̃��[�h�Ǝ����̍쐬
    3.���f���̒�`
    4.���f���̊w�K
    5.val_loss�ŏ����f���̃��[�h
    6.�e�X�g�̎��s
    7.���ʂ܂Ƃ߂̏o��

'''

from __future__ import print_function
from keras.models import Sequential, Model
from keras.models import model_from_json
from keras.layers import Dense, Activation, Input, Embedding
from keras.layers import LSTM
from keras.layers import add, concatenate, multiply
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
matplotlib.use('Agg')    #�����pyplot����ɏ������ƂŃT�[�o�ł������悤�ɂ��Ă���
import matplotlib.pyplot as plt
import subprocess

#----- �O���[�o���ϐ��ꗗ -----
my_epoch=100
vec_size=100
maxlen_words = 5
KeyError_set=set()
today_str=''
tmp_vec_dict=dict()


#----- �֐��Q -----

#���ԕ\��
def print_time(str1):
    today=datetime.datetime.today()
    print(str1)
    print(today)
    return today

#�w�K�f�[�^��e�X�g�f�[�^�ւ̑O����
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


#list�̊e�v�f��P��ŘA������string�^�ŕԂ�
def list_to_sent(list_line, start, end):
    sent=' '.join(list_line[start:end])
    return sent


#�w�K�f�[�^�ւ̑O�������s��
#���������C�A���t�@�x�b�g�ȊO�̕����̍폜�C1���P�ꂲ�Ƃɕ���
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
                    #���̑O������text8�Ƃ��̑O�����Ɠ���
                    line=preprocess_line(line)
                    line_list=line.split(' ')
                    line_len=len(line_list)
                    #max_len�ȉ��̎��͘A�����Ď���
                    if(text_len+line_len <= max_len):
                        if(text_len==0):
                            text=line
                        else:
                            text=text+' '+line
                        text_len=text_len+line_len
                    #max_len��蒷���Ƃ���max_len�P�ꂲ�Ƃɋ�؂��ăt�@�C���֏�������
                    else:
                        while (line_len>max_len):
                            if(text_len==0):
                                text=list_to_sent(line_list,0,max_len)
                            else:
                                text=text+' '+list_to_sent(line_list,0,max_len-text_len)
                            f_out.write(text+'\n')
                            text=''
                            text_len=0
                            #�c��̍X�V
                            line_list=line_list[max_len-text_len+1:]
                            line_len=len(line_list)
                        #while �I���i1�s�̖����̏����j
                        #�]��͎��̍s�ƘA��
                        text=list_to_sent(line_list,0,line_len)
                        text_len=line_len
                #for�I���i�t�@�C���̍Ō�̍s�̏����j
                if text_len!=0:
                    text=preprocess_line(text)
                    f_out.write(text+'\n')
                print('total '+str(i)+' line\n')
                print_time('preprpcess end')

    return new_path


#fasttext�̃x�N�g���t�@�C������P�ꎫ���ƃx�N�g�������̍쐬
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
                #�擪�s�ɂ͒P�ꐔ�Ǝ�������������Ă���̂Ŗ���
                line=line.replace('\n', '').replace('\r','')
                if line[-1]==' ':
                    line=line[:-1]
                tmp_list=line.split(' ')
                word=tmp_list[0]
                str_list=tmp_list[1:]
                #�����̍쐬
                #0�Ԗڂ̓p�f�B���O�p�̐����Ȃ̂Ŏg��Ȃ����Ƃɒ���
                word_indices[word]=i
                indices_word[i]=word
                vec_dict[word]=np.array(str_list, dtype=np.float32)
            i+=1

    word_indices['#OTHER']=i
    indices_word[i]='#OTHER'
    len_words=i
    return len_words, word_indices, indices_word, vec_dict


#fasttext�̃x�N�g���𓾂�
#���m��̏ꍇ�ɂ�fasttext�̃��f���Ăяo���Ď��s
#���m��͏W���Ɋi�[���C���ƂŃt�@�C���o��
def get_ft_vec(word, vec_dict, ft_path, bin_path):
    if word in vec_dict:
        return vec_dict[word]
    elif word in tmp_vec_dict:
        return tmp_vec_dict[word]
    else:
        KeyError_set.add(word)    #�v�f��ǉ�
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


# ���f���̍\�z
def build_model(len_words, embedding_matrix):
    f_input=Input(shape=(maxlen_words,))
    f_emb=Embedding(output_dim=vec_size, input_dim=len_words+1, input_length=maxlen_words, mask_zero=True, weights=[embedding_matrix], trainable=False)(f_input)

    f_layer=LSTM(128)(f_emb)

    r_input=Input(shape=(maxlen_words,))
    r_emb=Embedding(output_dim=vec_size, input_dim=len_words+1, input_length=maxlen_words, mask_zero=True, weights=[embedding_matrix], trainable=False)(r_input)
    r_layer=LSTM(128)(r_emb)

    merged_layer=add([f_layer, r_layer])

    out_full=Dense(vec_size,activation='relu')(merged_layer)
    out_layer=Dense(vec_size,activation='relu')(out_full)

    my_model=Model([f_input, r_input], out_layer)

    optimizer = RMSprop()
    my_model.compile(loss='mean_squared_error', optimizer=optimizer)

    return my_model


#�P�ꂩ�玫��ID��Ԃ�
def search_word_indices(word, word_to_id):
    if word in word_to_id:
        return word_to_id[word]
    else:
        return word_to_id['#OTHER']


#1�s�̕�������w�K�f�[�^�̌`���ɕϊ�
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
            r_sentences.append(r[::-1]) #�t���̃��X�g
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


#loss, val_loss�̒ǉ��X�V
def conect_hist(list_loss, list_val_loss, new_history):
    list_loss.extend(new_history.history['loss'])
    list_val_loss.extend(new_history.history['val_loss'])


#1�s10000�P��܂ł̃t�@�C������1�s����1��w�K����
#loss��val_loss�͊e�s�̊w�K���ʂ̒����l��Ԃ�
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
    with open(today_str+'loss.txt', 'a') as f_loss:
            f_loss.write('loss='+str(loss)+'  , val_loss='+str(val_loss)+'\n')
    return loss, val_loss


#my_epoch�̐������w�K�����肩����
def model_fit_loop(train_path, my_model, len_words, word_to_id, vec_dict, ft_path, bin_path):
    list_loss=list()
    list_val_loss=list()
    for ep_i in range(my_epoch):
        print('\nEPOCH='+str(ep_i+1)+'/'+str(my_epoch)+'\n')
        loss, val_loss=model_fit_once(train_path, my_model,len_words, word_to_id, vec_dict, ft_path, bin_path)
        list_loss.append(loss)
        list_val_loss.append(val_loss)

        #���f���̕ۑ�
        dir_name=today_str+'Model_'+str(ep_i+1)
        os.mkdir(dir_name)

        model_json_str = my_model.to_json()
        file_model=dir_name+'/my_model'
        open(file_model+'.json', 'w').write(model_json_str)
        my_model.save_weights(file_model+'.h5')


    return list_loss, list_val_loss


# �����̗������v���b�g
def plot_loss(list_loss, list_val_loss, title='model loss'):
    plt.plot(list_loss, color='blue', marker='o', label='loss')
    plt.plot(list_val_loss, color='green', marker='o', label='val_loss')
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(today_str+'loss_graph.png')
    #plt.show()�͍s���ƃv���O�������f����邩��NG


#�e�X�g�f�[�^�̑O����
def prepare_test(test_path, ch_path):

    th_len =maxlen_words/2    #�e�X�g�̍ۂ̒�����臒l
    test_f_sentences = list()
    test_r_sentences = list()

    #�e�X�g�f�[�^�ւ̓ǂݍ��݂ƑO����
    #�e�X�g�f�[�^�͊w�K�f�[�^�ƈقȂ�e�ʑ傫���Ȃ��̂ň�C�ɓǂݍ���ł���
    #�e�X�g�f�[�^��1�s1���1�s��<>��1�̂�
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
                    #�e�X�g�Ώۂ̃f�[�^�̓����ƑI���������X�g�Ɋi�[
                    ans_list.append(ans)
                    choi=tmp_choi[:-5]  #�����̃V���[�v�Ƃ����폜
                    ch_list.append(choi)
                    #�e�X�g�ΏۂƂȂ�f�[�^�݂̂��o��
                    with open(today_str+'testdata.txt', 'a') as data:
                        data.write(line+'\n')
        line_num+=1

    return test_f_sentences, test_r_sentences, ans_list, ch_list



#�e�X�g�f�[�^�̃x�N�g����
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


#2�̃x�N�g���̃R�T�C���ގ��x��Ԃ�
def calc_similarity(pred_vec, ans_vec):
    len_p=np.linalg.norm(pred_vec)
    len_a=np.linalg.norm(ans_vec)
    if len_p==0 or len_a==0:
        return 0.0
    return np.dot(pred_vec/len_p, ans_vec/len_a)


#�S�P��̒�����x�N�g���̗ގ��x�̍������Ƀt�@�C���o�́i���Ƃōl�@�p�j���C
#���1��Ƃ��̗ގ��x�C�I�����̊e��̏��ʂƗގ��x��Ԃ�
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
    list_ch = sorted(dict_ch.items(), key=lambda x: x[1],reverse=True)
    list_all = sorted(dict_all.items(), key=lambda x: x[1],reverse=True)
    with open(fname, 'a') as file:
        for w,sim in list_all:
            str=w+ ' ### '
            file.write(str)
        file.write('\n')
    return (list_all[0], list_ch)
    #�Ԃ�l��(�P��, �ގ��x), {(�P��, �ގ��x),(�P��, �ގ��x)...}


#�P��ƃ����N���X�g����P��̏��ʂ�string�^�ŕԂ�
def word_to_rank(word, ra_list):
    str_num=''
    if word in ra_list:
        str_num=str(ra_list.index(word))
    else:
        #�����Ƃ���-1
        str_num='-1'

    return str_num


#�S�P��̓��ގ��x1��̐�����t�@�C���������݁C���댋�ʂ�Ԃ�
def calc_rank1word(top, ans, ra_list, ans_sim):
    rank_top=word_to_rank(top[0], ra_list)
    rank_ans=word_to_rank(ans, ra_list)
    out=''
    with open(today_str+'rankOK.txt', 'a') as rOK:
        with open(today_str+'rankNG.txt', 'a') as rNG:
            out='pred= ('+top[0]+', '+rank_top+', '+str(top[1])+')   '+'ans= ('+ans+', '+rank_ans+', '+str(ans_sim)+')\n'
            if top[0]==ans:
                rOK.write(out)
                OK_num=1
            else:
                rNG.write(out)
                OK_num=0
    return OK_num


#�I�����őI�񂾍ۂ̐�����t�@�C���������݁C���댋�ʂ�Ԃ�
#choices=[(�P��,�ގ��x),(�P��,�ގ��x), ...]�̌`��
def calc_rank4choices(choices, ans, ra_list, ans_sim):
    pred=choices[0][0]
    rank_ans=word_to_rank(ans, ra_list)
    out='ans= ('+ans+', '+rank_ans+', '+str(ans_sim)+')    '+'choices='
    for word,sim in choices:
        out=out+'('+word+', '+word_to_rank(word, ra_list)+', '+str(sim)+')  '
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


#���𗦂̌v�Z���ʂ𕶎���ŕԂ�
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


#�e�X�g
def model_test(model, test_path, ch_path, word_to_id, vec_dict, ft_path, bin_path, id_to_word):
    #�e�X�g�f�[�^�̑O����
    f_sent, r_sent, ans_list, ch_list = prepare_test(test_path, ch_path)
    sent_num=len(f_sent)
    preds_list=list()
    top_list=list()
    choice_list=list()
    #�e�X�g�̎��s
    for i in range(sent_num):
        f_testX, r_testX = make_test_data(f_sent[i], r_sent[i], word_to_id)
        preds = min_model.predict([f_testX, r_testX], verbose=0)
        preds_list.append(preds)
        #�\�����ʂ̊i�[
        rank_file=today_str+'rank.txt'
        tmp=print_and_get_rank(preds, ch_list[i], rank_file, vec_dict, ft_path, bin_path, id_to_word)
        top=tmp[0]
        choice=tmp[1]
        top_list.append(top)
        choice_list.append(choice)
    #���𗦂̌v�Z�C�t�@�C���o��
    result_str=calc_acc(rank_file, ans_list, preds_list, top_list, choice_list, vec_dict, ft_path, bin_path)

    return result_str


#model.summary()�̃t�@�C���o�͗p
def myprint(s):
    with open(today_str+'model_summary.txt','a') as f:
        print(s, file=f)








#----- ������main���݂����Ȃ� -----

# 0.���낢��O����
#�J�n�����̃v�����g
start_time=print_time('all start')
start_time_str = start_time.strftime('%Y_%m_%d_%H%M')

#���f���Ƃ����ʂƂ����i�[����f�B���N�g���̍쐬
today_str=start_time_str+'epoch'+str(my_epoch)
program_name=os.path.basename(__file__)
settings=program_name[program_name.find('_e'):-3]
today_str=today_str+settings
if os.path.exists(today_str)==False:
    os.mkdir(today_str)
today_str='./'+today_str+'/'

#�w�K�f�[�^�̌��
train_big='../corpus/WikiSentWithEndMark1.txt'   # ��5.8GB�C��2000���s
train_enwiki='../corpus/enwiki.txt'   # ��24GB�C1�s�̂݁C��435���P��(��237���)
train_mid='../corpus/miniWiki_tmp8.txt'   # ��1.5MB�C��5000�s
train_small='../corpus/nietzsche.txt'   # ��600KB�C��1���s
train_test='../corpus/mini_text8.txt'

train_text8='../corpus/text8.txt'   # ��95MB 1�s�̂�, ��1700���P��(��7�����)  http://mattmahoney.net/dc/text8.zip



# 1.�w�K�f�[�^�̑O�����Ȃ�
tmp_path = train_text8     #�g�p����w�K�f�[�^
print('Loading  '+tmp_path)
train_path=preprocess(tmp_path)



# 2.fasttext�̃��[�h�Ǝ����̍쐬
'''
https://github.com/facebookresearch/fastText
����fastext�����O�Ɏ��s���Ă����C���̌��ʂ𗘗p
'''
ft_path='../../FastText/fastText-0.1.0/fasttext'

#�x�N�g���t�@�C���̌��
vec_enwiki='../../FastText/Model/enwiki_dim'+str(vec_size)+'_minC0.vec'
bin_enwiki='../../FastText/Model/enwiki_dim'+str(vec_size)+'_minC0.bin'
vec_text8='../../FastText/Model/text8_dim'+str(vec_size)+'_minC0.vec'
bin_text8='../../FastText/Model/text8_dim'+str(vec_size)+'_minC0.bin'

#���ۂɎg������
vec_path=vec_text8
bin_path=bin_text8

len_words, word_to_id, id_to_word, vec_dict=vec_to_dict(vec_path)

#embedding�ŗp����C�P�ꂩ��s��ւ̕ϊ��s��
embedding_matrix = np.zeros((len_words+1, vec_size))
for i in range(len_words):
    if i!=0:
        #ID��0�̒P�ꂪ���݂��Ȃ��̂�0�͔�΂�
        embedding_matrix[i] = get_ft_vec(id_to_word[i], vec_dict, ft_path, bin_path)


end_data=print_time('prepare data and fasttext end')



# 3.���f���̒�`
my_model=build_model(len_words, embedding_matrix)



# 4.���f���̊w�K
loss, val_loss=model_fit_loop(train_path, my_model, len_words, word_to_id, vec_dict, ft_path, bin_path)
plot_loss(loss, val_loss)

end_train=print_time('train end')



# 5.val_loss�ŏ����f���̃��[�h
min_i=np.array(val_loss).argmin()

min_model_file=today_str+'Model_'+str(min_i+1)+'/my_model.json'
min_weight_file=today_str+'Model_'+str(min_i+1)+'/my_model.h5'
print('Loading  '+min_model_file)

json_string = open(min_model_file).read()
min_model = model_from_json(json_string)
min_model.load_weights(min_weight_file)
optimizer = RMSprop()
min_model.compile(loss='categorical_crossentropy', optimizer=optimizer)

#min_model.summary(print_fn=myprint)
#summary���t�@�C���o�͂��������ǂ���͂��܂������Ȃ��悤��
plot_model(min_model, to_file=today_str+'model.png', show_shapes=True)

end_load=print_time('Load min_model end')



# 6.�e�X�g�̎��s
test_path = '../corpus/ans_all2000_2016.txt'     #�������e�X�g�f�[�^
ch_path= '../corpus/choi_all2000_2016.txt'     #�I�������e�X�g�f�[�^

result=model_test(min_model, test_path, ch_path, word_to_id, vec_dict, ft_path, bin_path, id_to_word)
print(result)

with open(today_str+'keyerror_words.txt', 'w') as f_key:
    for word in KeyError_set:
        f_key.write(word+'\n')


end_test=print_time('test end')



#7.���s���ʂ܂Ƃ߂̃t�@�C����������
#���L���e���t�@�C���ɂ܂Ƃ߂ďo��
'''
�E���s�����v���O������
�E���{�����i�J�n�����j
�E�ǂݍ��񂾊w�K�f�[�^
�E�P�ꐔ
�E�S�w�K��
�Eval_loss�ŏ��̊w�K��

�E�e�X�g����

�Emodel�̊T�v

�E�w�K�f�[�^�̑O�����C�����̍쐬�ɂɂ�����������
�Efasttext�̃��[�h��embedding_matrix�̍쐬�ɂ�����������
�E�w�K�ɂ����������ԁi�x�N�g�������܂ށj
�Eval_loss�ŏ����f���̃��[�h�ɂ�����������
�E�e�X�g�ɂ����������ԁi�x�N�g�����C���𗦂Ƃ����܂ށj
�E�S���v������������
'''

#���s���ʂ̂��ꂱ����t�@�C����������
min_model.summary()

with open(today_str+'summary.txt', 'a') as f:
    f.write('Result of '+program_name+'\n\n')

    f.write('start_time = '+ start_time_str+'\n')
    f.write('epoch = '+str(my_epoch)+'\n')
    f.write('train_data = '+ train_path+'\n')
    f.write('kind of words ='+str(len_words)+'\n')
    f.write('min_model = '+ min_model_file+'\n\n')

    f.write('result\n'+ result+'\n')

    f.write('TIME prepare data and fasttext= '+ str(end_data-start_time)+'\n')
    f.write('TIME train = '+ str(end_train-end_data)+'\n')
    f.write('TIME load min_model = '+ str(end_load-end_train)+'\n')
    f.write('TIME test = '+ str(end_test-end_load)+'\n\n')

    end_time=print_time('all end')
    f.write('TIME total = '+ str(end_time-start_time)+'\n')
