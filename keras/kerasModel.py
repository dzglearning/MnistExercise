# -*- coding: utf-8 -*-
"""
Created on Thu May  2 18:45:09 2019
@author: <dzg>

@software: spyder
"""
from keras.models import Sequential
from keras.layers import Dense,Activation
import tensorflow as tf
#import tensorflow.gfile as gfile
import os.path
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.datasets import mnist
from keras.utils import np_utils




from sklearn import preprocessing
from sklearn import metrics
import kerasdatapre as kdp

batchsize = 128
epoch = 5
node = 512

DataPath = 'D:\\dzg_project\\MnistExercise\\mnistdata\\'
SaveModelPath = 'D:\\dzg_project\\MnistExercise\\keras\\model\\'
ModelName = 'keras_mnist.h5'
SavePic =  'D:\\dzg_project\\MnistExercise\\keras\\images\\'
# 路径都要存在

def Kmodel(xtr,ytr,xte,yte):
    # 声明模型
    model = Sequential()
    model.add(Dense(node,input_shape=(784,)))
    model.add(Activation('relu'))
    model.add(Dense(node))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('softmax')) # 这里需要 keras<=2.1.5 版本

    # 编译模型
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    # 训练模型  verbose = 2 为每个epoch输出一行记录
    history = model.fit(xtr, ytr, batch_size=batchsize, epochs=epoch,
                    verbose=2, validation_data=(xte, yte))
    # 保存模型
    if tf.gfile.Exists(SaveModelPath):
        tf.gfile.DeleteRecursively(SaveModelPath)
    tf.gfile.MakeDirs(SaveModelPath)
    model.save(os.path.join(SaveModelPath,ModelName))
    print('Model has Saved at:{}'.format(os.path.join(SaveModelPath,ModelName)))        
    
    return history


# 评估测试模型  注意这里输入的ytest，有经过onehot和没经onehot编码的两种
def testmodel(modelpath,modelname,xte,ytecode,ytenocode):
    Mpath = os.path.join(modelpath,modelname)
    mnistmodel = load_model(Mpath)
    loss_and_metrics = mnistmodel.evaluate(xte, ytecode, verbose=2)
    print("Test Loss: {}".format(loss_and_metrics[0]))
    print("Test Accuracy: {}%".format(loss_and_metrics[1]*100))
    
    predicted_classes = mnistmodel.predict_classes(xte)
    
    correct_indices = np.nonzero(predicted_classes == ytenocode)[0]
    incorrect_indices = np.nonzero(predicted_classes != ytenocode)[0]
    print("Classified correctly count: {}".format(len(correct_indices)))
    print("Classified incorrectly count: {}".format(len(incorrect_indices)))


# 显示 loss 和准确率 指标
def plotloss(model):
    plt.figure()
#    plt.suptitle('Loss and Accuracy')
    plt.subplot(2,1,1)
    plt.plot(model.history['acc'])
    plt.plot(model.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')
    
    plt.subplot(2,1,2)
    plt.plot(model.history['loss'])
    plt.plot(model.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.tight_layout()
    plt.savefig(SavePic+'LossAndAccuracy.png')
    plt.show()
        

# 主函数
if __name__ == '__main__':
    x = kdp.loaddata(DataPath)
    kdp.distribution(x[1],x[3])
    xtrnor,xtenor = kdp.normaldata(x[0],x[2])
    ytrcode,ytecode = kdp.onehot(x[1],x[3])
    hismodel = Kmodel(xtrnor,ytrcode,xtenor,ytecode)
    plotloss(hismodel)
    testmodel(SaveModelPath,ModelName,xtenor,ytecode,x[3])