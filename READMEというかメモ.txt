あぷろだ代わり

keras使ってRNNの勉強というか練習する過程のいろいろ

python ○○.py 2>&1 | tee ○○○.txt

today=datetime.datetime.today()
today_str = today.strftime("%Y_%m_%d_%H_%M_%S")

filename2=filename+today_str+".txt" 
みたいな

cygwin/home/download
にサーバから結果ダウンロードしてある


モデルの可視化
from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_char_word.png')

https://pondad.net/ai/2016/12/25/keras-mnist.html
