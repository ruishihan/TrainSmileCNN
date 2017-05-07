#coding: utf-8

#train_smile.py
#Created: 2015-16
#Updated: 2017/05/07

#CNNで笑顔認識
#笑顔認識実行ファイル

#【参考】深層学習でアニメ顔を分類する
#http://qiita.com/hogefugabar/items/312707a09d29632e7288

from smile_cnn import CNN
from smile_data import FaceDataset
from chainer import cuda

#データセットの読み込み
print 'load FaceImage dataset'
dataset = FaceDataset()
dataset.load_data_target()

data = dataset.data
target = dataset.target

#出力層のユニット数(分類クラス数)
n_outputs = dataset.get_n_types_target()

#モデル構築
cnn = CNN(data=data,
          target=target,
          gpu=0,
          n_outputs=n_outputs)

#学習とテスト
cnn.train_and_test(n_epoch=100) #学習回数100回
