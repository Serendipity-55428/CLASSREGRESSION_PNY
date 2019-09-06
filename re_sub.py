#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@contact: 1243049371@qq.com
@software: Pycharm
@file: re_sub
@time: 2019/9/4 下午10:17
'''
import tensorflow as tf
import numpy as np
from cl25 import spliting, LoadFile, SaveFile, input
def acc_regression(Threshold, y_true, y_pred):
    '''
    回归精确度（预测值与实际值残差在阈值范围内的数量/样本总数）
    :param Threshold: 预测值与实际值之间的绝对值之差阈值
    :param y_true: 样本实际标签
    :param y_pred: 样本预测结果
    :return: 精确率，type: ndarray
    '''
    # 残差布尔向量
    is_true = np.abs(y_pred - y_true) <= Threshold
    is_true_cast = np.cast(is_true, tf.float32)
    acc_rate_regression = np.sum(is_true_cast) / is_true_cast.shape[0]
    return acc_rate_regression

class R_regression(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # 卷积层
        self.__reshape1 = tf.keras.layers.Reshape(target_shape=[10, 10, 1], name='reshape')
        self.__conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=[5, 5], padding='same', activation=tf.nn.relu,
                                              kernel_initializer=tf.keras.initializers.TruncatedNormal, name='conv1')
        self.__pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same', name='pool1')
        self.__bn1 = tf.keras.layers.BatchNormalization(name='bn1')
        self.__conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], padding='same', activation=tf.nn.relu,
                                              kernel_initializer=tf.keras.initializers.TruncatedNormal, name='conv2')
        self.__pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same', name='pool2')
        self.__flat1 = tf.keras.layers.Flatten(name='flat1')
        #rnn层
        self.__reshape2 = tf.keras.layers.Reshape(target_shape=[24, 24], name='x_lstm')
        self.__lstm1 = tf.keras.layers.LSTM(units=128, dropout=0.8, return_sequences=True, name='lstm1')
        self.__lstm2 = tf.keras.layers.LSTM(units=128, dropout=0.8, return_sequences=False, name='lstm2')
        self.__flat2 = tf.keras.layers.Flatten(name='flat2')
        #dnn层
        def concat(inputs):
            return tf.concat(values=inputs, axis=1)
        self.__concat = tf.keras.layers.Lambda(concat, name='concat1')
        self.__fc1 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu, use_bias=True,
                                           kernel_initializer=tf.keras.initializers.TruncatedNormal,
                                           bias_initializer=tf.keras.initializers.TruncatedNormal, name='x_fc1')
        self.__dropout1 = tf.keras.layers.Dropout(rate=0.2, name='x_dpt1')
        self.__fc2 = tf.keras.layers.Dense(units=200, activation=tf.nn.relu, use_bias=True,
                                           kernel_initializer=tf.keras.initializers.TruncatedNormal,
                                           bias_initializer=tf.keras.initializers.TruncatedNormal, name='x_fc2')
        self.__dropout2 = tf.keras.layers.Dropout(rate=0.2, name='x_dpt2')
        self.__output = tf.keras.layers.Dense(units=1, activation=tf.nn.relu, use_bias=True,
                                              kernel_initializer=tf.keras.initializers.TruncatedNormal,
                                              bias_initializer=tf.keras.initializers.TruncatedNormal, name='x_fc3')
    def call(self, inputs, training=True, mask=None):
        layer = self.__reshape1(inputs[-1])
        layer = self.__conv1(layer)
        layer = self.__pool1(layer)
        layer = self.__bn1(layer)
        layer = self.__conv2(layer)
        layer = self.__pool2(layer)
        layer = self.__flat1(layer)
        layer = self.__reshape2(layer)
        layer = self.__lstm1(layer)
        layer = self.__lstm2(layer)
        layer = self.__flat2(layer)
        layer = self.__concat(inputs=[inputs[0], layer])
        layer = self.__fc1(layer)
        layer = self.__dropout1(layer)
        layer = self.__fc2(layer)
        layer = self.__dropout2(layer)
        return self.__output(layer)

def graph_re(dataset, save_path):
    ''''''
    r_regression = R_regression()
    optimizer = tf.keras.optimizers.SGD(lr=1e-2)
    r_regression.compile(optimizer=optimizer, loss=tf.keras.losses.mean_squared_error)
    train_data, test_data = spliting(dataset, 6000)
    flag = 0
    for  epoch in range(10000):
        for train_data_batch in input(dataset=train_data, batch_size=500):
            loss_train, _ = r_regression.train_on_batch(x=[train_data_batch[:, :4], train_data_batch[:, 4:-1]],
                                                      y=train_data_batch[:, -1])
            if epoch % 100 == 0 and flag == 0:
                print('第%s轮后训练集损失函数值为: %s' % (epoch, loss_train))
                flag = 1
        if epoch % 100 == 0:
            r_predict = r_regression.predict(x=[test_data[:, :4], test_data[:, 4:-1]], verbose=0)
            acc = acc_regression(Threshold=10, y_true=test_data[:, -1], y_pred=r_predict)
            print('测试集准确率为: %s' % acc)
        flag = 0
    r_regression.save(save_path)

if __name__ == '__main__':
    path = '/home/xiaosong/桌面/pny_regression_sub.pickle'
    save_path = '/home/xiaosong/桌面/graph_cl_re/graph_re.h5'
    dataset = LoadFile(p=path)
    graph_re(dataset=dataset, save_path=save_path)
