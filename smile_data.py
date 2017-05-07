#coding: utf-8

#smile_data.py
#Created: 2015-16
#Updated: 2017/05/07

#CNNで笑顔認識
#データセット読み込み(フォルダ名ラベル化)

#【参考】深層学習でアニメ顔を分類する
#http://qiita.com/hogefugabar/items/312707a09d29632e7288
#animeface.py(データセット前処理)

import os
import six.moves.cPickle as pickle
import numpy as np
import cv2 as cv

class FaceDataset:
    def __init__(self):
        
        #データセットが格納されているフォルダのパス
        #12,000枚のデータセットを読み込む場合(1,000枚と6,000枚のフォルダも選択可:ここを書き換える)
        self.data_dir_path = u"./FaceImages/FaceDataset_x12000/"

        self.data = None
        self.target = None
        self.n_types_target = -1
        self.dump_name = u'dataset'

        #入力画像サイズ:32*32
        self.image_size = 32

    #フォルダリストの取得
    def get_dir_list(self):
        tmp = os.listdir(self.data_dir_path)
        if tmp is None:
            return None
        return sorted([x for x in tmp if os.path.isdir(self.data_dir_path+x)])

    #フォルダ名をラベルとして読み込む
    def get_class_id(self, fname):
        dir_list = self.get_dir_list()
        dir_name = filter(lambda x: x in fname, dir_list)
        return dir_list.index(dir_name[0])

    #データの読み込み
    def load_data_target(self):
        if os.path.exists(self.dump_name):
            self.load_dataset()
        if self.target is None:
            dir_list = self.get_dir_list()
            ret = {}
            self.target = []
            target_name = []
            self.data = []

            #フォルダリストの読み出し
            for dir_name in dir_list:
                file_list = os.listdir(self.data_dir_path+dir_name)

                #ファイルリストの読み出し
                for file_name in file_list:
                    root, ext = os.path.splitext(file_name)
                    if ext == u'.jpg':  #ファイル名に".jpg"が含まれるものを抽出
                        abs_name = self.data_dir_path+dir_name+'/'+file_name
                        
                        #ラベルデータの読み込み as target
                        class_id = self.get_class_id(abs_name)
                        self.target.append(class_id)
                        target_name.append(str(dir_name))
                        
                        #画像データの読み込み as data
                        image = cv.imread(abs_name)

                        #リサイズ(32*32)
                        image = cv.resize(image, (self.image_size, self.image_size))

                        image = image.transpose(2,0,1)  #[0]channel,[1]row,[2]colum
                        self.data.append(image)

            self.index2name = {}
            for i in xrange(len(self.target)):
                self.index2name[self.target[i]] = target_name[i]

        #numpy配列への変換(NNに読み込める形式)
        self.data = np.array(self.data, np.float32)
        self.target = np.array(self.target, np.int32)

        self.dump_dataset()

    def get_n_types_target(self):
        if self.target is None:
            self.load_data_target()

        if self.n_types_target is not -1:
            return self.n_types_target

        tmp = {}
        for target in self.target:
            tmp[target] = 0
        return len(tmp)

    #構築したNN用データセットの保存
    def dump_dataset(self):
        pickle.dump((self.data,self.target,self.index2name), open(self.dump_name, 'wb'), -1)

    #構築したNN用データセットの読み込み
    def load_dataset(self):
        self.data, self.target, self.index2name = pickle.load(open(self.dump_name, 'rb'))
