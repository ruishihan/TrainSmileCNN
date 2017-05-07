#coding: utf-8

#smile_cnn.py
#Created: 2015-16
#Updated: 2017/05/07

#CNNで笑顔認識
#モデルと伝搬規則の定義,学習とテスト

#【参考】深層学習でアニメ顔を分類する
#http://qiita.com/hogefugabar/items/312707a09d29632e7288
#CNN.py

#【参考】人工知能に関する断創録-Chainerによる畳み込みニューラルネットワークの実装
#http://aidiary.hatenablog.com/entry/20151007/1444223445

import time
import numpy as np
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions as F
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
import six.moves.cPickle as pickle

#モデルと伝搬規則の定義
class ImageNet(FunctionSet):

    #モデル(CNN)の定義
    def __init__(self, n_outputs):
        super(ImageNet, self).__init__(
            conv1=  F.Convolution2D(3, 20, 5),  #1つめの畳み込み:入力3枚,出力20枚,フィルタサイズ5
            conv2=  F.Convolution2D(20, 50, 5), #2つ目の畳み込み:入力20枚,出力50枚,フィルタサイズ5
            l3=     F.Linear(800, 500),         #MLP:入力800ユニット,出力500ユニット
            l4=     F.Linear(500, n_outputs)    #出力層:入力500ユニット,出力は分類クラス数と同数
        )

    #伝搬規則の定義
    def forward(self, x_data, y_data, train=True, gpu=-1):

        if gpu >= 0:
            x_data = cuda.to_gpu(x_data)
            y_data = cuda.to_gpu(y_data)

        x, t = Variable(x_data), Variable(y_data)
        h = F.max_pooling_2d(F.relu(self.conv1(x)), ksize=2, stride=2)  #1つ目のプーリング,活性化関数ReLU
        h = F.max_pooling_2d(F.relu(self.conv2(h)), ksize=3, stride=3)  #2つ目のプーリング,活性化関数ReLU
        h = F.dropout(F.relu(self.l3(h)), train=train)                  #Dropout(過学習制御)
        y = self.l4(h)

        #交差エントロピーによる誤差の計算と，精度の計算
        return F.softmax_cross_entropy(y, t), F.accuracy(y,t)

#学習とテスト
class CNN:
    def __init__(self, data, target, n_outputs, gpu=-1):

        self.model = ImageNet(n_outputs)
        self.model_name = 'cnn_model'

        if gpu >= 0:
            self.model.to_gpu()

        self.gpu = gpu

        #学習データとテストデータに分割
        self.x_train,\
        self.x_test,\
        self.y_train,\
        self.y_test = train_test_split(data, target, test_size=0.1)

        self.n_train = len(self.y_train)
        self.n_test = len(self.y_test)

        #Optimizerの設定
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model.collect_parameters())

    def predict(self, x_data, gpu=-1):
        return self.model.predict(x_data, gpu)

    #学習とテスト
    def train_and_test(self, n_epoch=100, batchsize=100):

        fp1 = open("train.txt", "w")
        fp2 = open("test.txt", "w")

        fp1.write("epoch\ttrain_loss\ttrain_accuracy\n")
        fp2.write("epoch\ttest_loss\ttest_accuracy\n")

        start_time = time.clock()   #時間計測開始(開始時刻取得)
        
        epoch = 1
        best_accuracy = 0
        while epoch <= n_epoch:
            print 'epoch', epoch

            perm = np.random.permutation(self.n_train)

            #学習
            sum_train_accuracy = 0
            sum_train_loss = 0
            for i in xrange(0, self.n_train, batchsize):
                x_batch = self.x_train[perm[i:i+batchsize]]
                y_batch = self.y_train[perm[i:i+batchsize]]

                real_batchsize = len(x_batch)

                self.optimizer.zero_grads()
                loss, acc = self.model.forward(x_batch, y_batch, train=True, gpu=self.gpu)
                loss.backward()
                self.optimizer.update()

                #学習データに対する誤差算出
                sum_train_loss += float(cuda.to_cpu(loss.data)) * real_batchsize

                #学習データに対する精度算出
                sum_train_accuracy += float(cuda.to_cpu(acc.data)) * real_batchsize

            #学習結果テキスト出力
            print 'train mean loss={}, accuracy={}'.format(sum_train_loss/self.n_train, sum_train_accuracy/self.n_train)

            #ファイル出力
            fp1.write("%d\t%f\t%f\n" % (epoch, sum_train_loss / self.n_train, sum_train_accuracy / self.n_train))
            fp1.flush()

            #テスト
            sum_test_accuracy = 0
            sum_test_loss = 0
            for i in xrange(0, self.n_test, batchsize):
                x_batch = self.x_test[i:i+batchsize]
                y_batch = self.y_test[i:i+batchsize]

                real_batchsize = len(x_batch)

                loss, acc = self.model.forward(x_batch, y_batch, train=False, gpu=self.gpu)

                #テストデータに対する誤差算出
                sum_test_loss += float(cuda.to_cpu(loss.data)) * real_batchsize

                #テストデータに対する精度算出
                sum_test_accuracy += float(cuda.to_cpu(acc.data)) * real_batchsize

            #テスト結果テキスト出力
            print 'test mean loss={}, accuracy={}'.format(sum_test_loss/self.n_test, sum_test_accuracy/self.n_test)

            #ファイル出力
            fp2.write("%d\t%f\t%f\n" % (epoch, sum_test_loss / self.n_test, sum_test_accuracy / self.n_test))
            fp2.flush()

            epoch += 1

        end_time = time.clock() #時間計測終了(終了時刻取得)

        #実行時間の表示(終了時刻-開始時刻)
        print('Execution time: %f[s]' % (end_time - start_time))

        fp1.close()
        fp2.close()

    #モデルの保存
    def dump_model(self):
        self.model.to_cpu()
        pickle.dump(self.model, open(self.model_name, 'wb'), -1)

    #モデルの読み込み
    def load_model(self):
        self.model = pickle.load(open(self.model_name,'rb'))
        if self.gpu >= 0:
            self.model.to_gpu()
        self.optimizer.setup(self.model.collect_parameters())
