# -*- coding: utf-8 -*-
"""
Created on Thu  22:17:30 2019
@author: <dzg>

@software: spyder
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

batchsize = 128            #每轮喂入网络图片数量
learning_rate = 0.01    #初始学习率
steps = 1000               #训练轮数
input_node = 784    # 神经网络输入节点，图片像素值784个点，一维数组
output_node = 10    # 输出十个数，每个数索引号对应数字出现的概率
layer_node = 500    # 隐藏层的节点
MODEL_SAVE_PATH="./model/"  #模型保存路径
MODEL_NAME="mnist_model"    #模型保存文件名


def get_weight(shape):
    w = tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    print("Got weights!\n")
    return w

def get_bias(shape):  
    b = tf.Variable(tf.zeros(shape))
    print("Got basis!\n")  
    return b
	
def forward(x):
    w1 = get_weight([784, 500])
    b1 = get_bias([500])
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1) # 第一层参数，偏置，输出

    w2 = get_weight([500, 10])
    b2 = get_bias([10])
    y = tf.matmul(y1, w2) + b2
    print("Completed forward compute!\n")
    # 要对输出使用softmax 函数，转化为概率分布，所以不过relu函数
    return y


def backward(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, input_node])
        y_ = tf.placeholder(tf.float32, [None, output_node])
        y = forward(x)       #调用前项传播函数计算输出y
        global_step = tf.Variable(0, trainable=False)   #轮数计数器初值，不可训练
    
        #调用包含正则化的损失函数
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
        loss = tf.reduce_mean(ce)
    #    loss = cem + tf.add_n(tf.get_collection('losses'))
    
        #定义训练过程
        train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    
        #实例化声明
        saver = tf.train.Saver()
    
        #初始化变量
        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
    #        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
    #        if ckpt and ckpt.model_checkpoint_path:            
    #            saver.restore(sess,ckpt.model_checkpoint_path)
            for i in range(steps):
                xs, ys = mnist.train.next_batch(batchsize)
                # _ 对应train_op 的输出，空出变量位，节省空间
                _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
                if i % 50 == 0:
                    print("经过 %d 轮训练, 在训练集中 loss 值为： %g." % (step, loss_value))
                    saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
                    #保存模型到当前会话


def test(mnist):
#    with tf.Graph().as_default() as g:  #复现计算图
#    with tf.get_default_graph() as g:
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, [None, input_node])
    y_ = tf.placeholder(tf.float32, [None, output_node])
    y = forward(x)      #前向传播计算输出y
    
    saver = tf.train.Saver()
		
    #计算准确率
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    with tf.Session() as sess:
        saver.restore(sess,tf.train.latest_checkpoint("./model/"))
        accuracy_score = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
        print('测试准确率：',accuracy_score)
        
def main():
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    backward(mnist)
    test(mnist)
    print("Completed  compute!\n")

if __name__ == '__main__':
    main()

