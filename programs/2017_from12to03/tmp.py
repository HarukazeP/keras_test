



























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
    with open(train_path) as f:
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

with open(today_str+'result.txt', 'a') as rslt:
    rslt.write(result+'\ntotal time ='+str(diff_time))

print(result)



print('all_end = ',end_time)
plot_loss(list_loss, list_val_loss)
plot_model(min_model, to_file=today_str+'model.png', show_shapes=True)



print('total =',diff_time)
