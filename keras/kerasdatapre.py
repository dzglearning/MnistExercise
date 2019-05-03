# -*- coding: utf-8 -*-
"""
author: dzg
software:spyder
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils
from sklearn import preprocessing
from sklearn import metrics


DataPath = 'D:\\dzg_project\\MnistExercise\\mnistdata\\'
SavePic =  'D:\\dzg_project\\MnistExercise\\keras\\images\\'
# 路径都要存在

# 加载数据
def loaddata(path): 
    (X_train, y_train), (X_test, y_test) = mnist.load_data(path + 'mnist.npz')
    print('X_train shape{} type:{}'.format(X_train.shape,type(X_train)))
    print('y_train shape{} type:{}'.format(y_train.shape,type(y_train)))
    print('X_test shape{} type:{}:'.format(X_test.shape,type(X_test)))
    print('y_test shape{} type:{}:'.format(y_test.shape,type(y_test)))
    print('Load data finished!\n\n\tfor examples!')
    fig = plt.figure(figsize=(6,6))
    fig.suptitle('Some Examples of X_train') # 总标题
    for i in range(16):
        plt.subplot(4,4,i+1)
#        plt.tight_layout() 自适应大小
        plt.imshow(X_train[i],cmap='Greys')
        plt.title('numbers:{}'.format(y_train[i]))
        plt.xticks([])
        plt.yticks([])  #隐藏坐标轴
    plt.savefig(SavePic+'XtrainExamples.png')
    return X_train,y_train,X_test,y_test


# 数据分布
def distribution(ytr,yte):
    ytrlabels, ytrcount = np.unique(ytr,return_counts=True)
    ytelabels, ytecount = np.unique(yte,return_counts=True)
    dislist = [(ytrlabels,ytrcount), (ytelabels,ytecount)]
    plt.figure(figsize=(6,10))
    plt.suptitle('Label Distribution')
    i = 1
    for label,count in dislist:
#        print(label,count)
        plt.subplot(2,1,i)
        i += 1
        plt.bar(label, count, width = 0.7, align='center')
        plt.xlabel("Label")
        plt.ylabel("Count")
        plt.xticks(label)
        for a,b in zip(label, count):       # 打包成一个元组 迭代器
            plt.text(a, b, '%d' % b, ha='center', va='bottom',fontsize=10)
    plt.savefig(SavePic+'Ydistribution.png')

# 数据规范化 
def normaldata(xtrold,xteold):
    Xtrain = xtrold.reshape(60000,784).astype('float32')
    Xtest = xteold.reshape(10000,784).astype('float32')
    scaler = preprocessing.MaxAbsScaler().fit(Xtrain)
    scaler = preprocessing.MaxAbsScaler().fit(Xtest)
    xtrnew = scaler.transform(Xtrain)
    xtenew = scaler.transform(Xtest)  #标准化 
#    xtrnew = Xtrain/255
#    Xtenew = Xtest/255  # 缩放 效果是一样的
    return xtrnew,xtenew


# one-hot编码
def onehot(ytr,yte):
    encode = preprocessing.LabelBinarizer()
    ytrcode = encode.fit_transform(ytr)
    ytecode = encode.fit_transform(yte)
#    n_classes = 10     # 同理的编码onrhot编码
#    ytrcode = np_utils.to_categorical(ytr,n_classes)
#    ytecode = np_utils.to_categorical(yte,n_classes)
    return ytrcode,ytecode
    
    
if __name__ == '__main__':
    x = loaddata(DataPath)
    distribution(x[1],x[3])
    normaldata(x[0],x[2])
    onehot(x[1],x[3])