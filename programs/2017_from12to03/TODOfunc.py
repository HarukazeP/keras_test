
#学習データへの前処理を行う
def preprocess(train_path):
	new_path=train_path[:-4]+'_preprpcessed.txt'
	if os.path.exists(new_path)==False:
		print('Preprpcessing training data...')
		#ここに処理
		#10000単語ごとに区切るやつ
		'''
		これ今までのやつ
		#学習データへの前処理

		with open(train_path) as f_in:
		    with open(tmp_path, 'w') as f_out:
		        for line in f_in:
		            text=line.lower()
		            text = re.sub(r"[^a-z ]", "", text)
		            text = re.sub(r"[ ]+", " ", text)
		            f_out.write(text)
		'''
	
	
	return new_path



#TODO すでに1行10000単語のデータ10000，リストに格納する関数も別つくる
#loss とかval_loss 平均する？
def model_fit(train_path, my_model):	#TODO この関数の作成
    with open(train_path) as f:
    	
        read_i=0
        text=""
        for line in f:
            read_i+=1
            t_line = t_line.replace("\n", " ").replace('\r','')
            t_line = re.sub(r"[ ]+", " ", t_line)
            text=text+' '+t_line
            # 1000行ごとに学習
            if(read_i % 1000==0):
                my_hist=model_fit(text, my_model)
                conect_hist(list_loss, list_val_loss, my_hist)
                text=""




	return loss, val_loss#TODO returnの仕方も要検討














