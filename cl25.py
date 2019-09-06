#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@contact: 1243049371@qq.com
@software: Pycharm
@file: cl25
@time: 2019/9/4 下午10:17
'''
from classifier_dataset import LoadFile, SaveFile
from collections import Counter
import tensorflow as tf
import numpy as np
import pandas as pd
class Resnet:

    def __init__(self, x, filters, kernel_size, name, padding='same', activation=tf.nn.relu,
                 kernel_initializer=tf.keras.initializers.TruncatedNormal):
        '''
        残差类属性初始化函数
        :param x: 待输入张量, Tensor/Variable
        :param filters: 卷积核个数, int
        :param kernel_size: 卷积核长宽尺寸, list
        :param name: 节点名, str
        :param padding: 标记是否自动补零, str
        :param activation: 激活函数, func
        :param kernel_initializer: 参数初始化函数, func
        '''
        self.__x = x
        self.__filters = filters
        self.__kernel_size = kernel_size
        self.__padding = padding
        self.__activation = activation
        self.__kernel_initializer = kernel_initializer
        self.__name = name

    def resnet_2layers(self):
        '''
        两层卷积的子网络结构
        :return: 子网络残差结构输出
        '''
        conv1 = tf.keras.layers.Conv2D(filters=self.__filters,
                                       kernel_size=self.__kernel_size,
                                       padding=self.__padding,
                                       activation=self.__activation,
                                       kernel_initializer=self.__kernel_initializer)(self.__x)
        conv2 = tf.keras.layers.Conv2D(filters=self.__filters,
                                       kernel_size=self.__kernel_size,
                                       padding=self.__padding,
                                       kernel_initializer=self.__kernel_initializer)(conv1)
        if self.__x.get_shape().as_list()[-1] != conv2.get_shape().as_list()[-1]:
            x = tf.keras.layers.Conv2D(filters=self.__filters,
                                       kernel_size=[1, 1],
                                       padding=self.__padding,
                                       kernel_initializer=self.__kernel_initializer)(self.__x)
        else:
            x = self.__x
        combination = tf.keras.layers.Add()([conv2, x])
        relu = tf.keras.layers.ReLU(name=self.__name)(combination)
        return relu

    def resnet_3layers(self):
        conv1 = tf.keras.layers.Conv2D(filters=self.__filters // 4,
                                       kernel_size=[1, 1],
                                       padding=self.__padding,
                                       activation=self.__activation,
                                       kernel_initializer=self.__kernel_initializer)(self.__x)
        conv2 = tf.keras.layers.Conv2D(filters=self.__filters // 4,
                                       kernel_size=self.__kernel_size,
                                       padding=self.__padding,
                                       kernel_initializer=self.__kernel_initializer)(conv1)
        conv3 = tf.keras.layers.Conv2D(filters=self.__filters,
                                       kernel_size=[1, 1],
                                       padding=self.__padding,
                                       kernel_initializer=self.__kernel_initializer)(conv2)
        if self.__x.get_shape().as_list()[-1] != conv2.get_shape().as_list()[-1]:
            x = tf.keras.layers.Conv2D(filters=self.__filters,
                                       kernel_size=[1, 1],
                                       padding=self.__padding,
                                       kernel_initializer=self.__kernel_initializer)(self.__x)
        else:
            x = self.__x
        combination = tf.keras.layers.Add()([conv3, x])
        relu = tf.keras.layers.ReLU(name=self.__name)(combination)
        return relu

def Cl25():
    ''''''
    with tf.name_scope('input'):
        input1 = tf.keras.Input(shape=(4,), name='input1')
        input2 = tf.keras.Input(shape=(100,), name='input2')
    with tf.name_scope('resnet'):
        layer = tf.keras.layers.Reshape(target_shape=(10, 10, 1), name='reshape1')(input2)
        # 初始卷积池化层
        layer = tf.keras.layers.Conv2D(filters=32, kernel_size=[5, 5], padding='same', activation=tf.nn.relu,
                                       kernel_initializer=tf.keras.initializers.TruncatedNormal, name='conv1')(layer)
        layer = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same', name='pool1')(layer)
        # 残差层
        resnet = Resnet(x=layer, filters=64, kernel_size=[3, 3], name='resnet1')
        res1 = resnet.resnet_2layers()

        resnet2 = Resnet(x=res1, filters=128, kernel_size=[3, 3], name='resnet2')
        res2 = resnet2.resnet_2layers()

        resnet3 = Resnet(x=res2, filters=256, kernel_size=[3, 3], name='resnet3')
        res3 = resnet3.resnet_2layers()

        resnet4 = Resnet(x=res3, filters=256, kernel_size=[3, 3], name='resnet4')
        res4 = resnet4.resnet_2layers()

        # resnet5 = Resnet(x=res4, filters=128, kernel_size=[3, 3], name='resnet5')
        # res5 = resnet5.resnet_2layers()
        flat = tf.keras.layers.Flatten(name='flat')(res4)
    with tf.name_scope('dnn'):
        def concat(inputs):
            return tf.concat(values=inputs, axis=1)
        layer = tf.keras.layers.Lambda(concat)(inputs=[flat, input1])
        layer = tf.keras.layers.Dense(units=100, activation=tf.nn.relu, use_bias=True,
                                      kernel_initializer=tf.keras.initializers.TruncatedNormal,
                                      bias_initializer=tf.keras.initializers.TruncatedNormal, name='x_fc1')(layer)
        layer = tf.keras.layers.Dense(units=200, activation=tf.nn.relu, use_bias=True,
                                      kernel_initializer=tf.keras.initializers.TruncatedNormal,
                                      bias_initializer=tf.keras.initializers.TruncatedNormal, name='x_fc2')(layer)
        layer = tf.keras.layers.Dense(units=25, activation=tf.nn.softmax, use_bias=True,
                                       kernel_initializer=tf.keras.initializers.TruncatedNormal,
                                       bias_initializer=tf.keras.initializers.TruncatedNormal, name='output')(layer)

    model_cl25 = tf.keras.Model(inputs=[input1, input2], outputs=layer)
    return model_cl25

def input(dataset, batch_size):
    '''
    按照指定批次大小随机输出训练集中一个批次的特征/标签矩阵
    :param dataset: 数据集特征矩阵(特征经过01编码后的)
    :param batch_size: 批次大小
    :return: 特征矩阵/标签
    '''
    for i in range(0, len(dataset), batch_size):
        yield dataset[i:i+batch_size, :]

def spliting(dataset, size):
    '''
    留一法划分训练和测试集
    :param dataset: 特征数据集/标签
    :param size: 测试集大小
    :return: 训练集和测试集特征矩阵/标签
    '''
    #随机得到size大小的交叉验证集
    test_row = np.random.randint(low=0, high=len(dataset)-1, size=(size))
    #从dataset中排除交叉验证集
    train_row = list(filter(lambda x: x not in test_row, range(len(dataset)-1)))
    return dataset[train_row, :], dataset[test_row, :]

def onehot(dataset, class_number):
    '''
    将所有标签按数值大小编码为one-hot稀疏向量
    :param dataset: 数据集特征矩阵,最后一列为半径标签
    :param class_number: onehot编码长度
    :return: 标签半径被编码后的数据集
    '''
    dataset_pd = pd.DataFrame(data=dataset, columns=[str(i) for i in range(dataset.shape[-1]-1)]+['cl'])
    dataset_onehot = np.array([0])
    for cl_number in range(class_number):
        dataset_sub_pd = dataset_pd.loc[dataset_pd['cl'] == cl_number]
        if dataset_sub_pd.values.shape[0] == 0: continue
        one_hot_label = np.zeros(shape=[dataset_sub_pd.values.shape[0], class_number], dtype=np.float32)
        one_hot_label[:, cl_number] = 1
        one_hot_dataset = np.hstack((dataset_sub_pd.values[:, :-1], one_hot_label))
        dataset_onehot = one_hot_dataset if dataset_onehot.any() == 0 else \
            np.vstack((dataset_onehot, one_hot_dataset))
    return dataset_onehot

# def acc_regression(Threshold, y_true, y_pred):
#     '''
#     回归精确度（预测值与实际值残差在阈值范围内的数量/样本总数）
#     :param Threshold: 预测值与实际值之间的绝对值之差阈值
#     :param y_true: 样本实际标签
#     :param y_pred: 样本预测结果
#     :return: 精确率，type: ndarray
#     '''
#     # 残差布尔向量
#     is_true = np.abs(y_pred - y_true) <= Threshold
#     is_true_cast = np.cast(is_true, tf.float32)
#     acc_rate_regression = np.sum(is_true_cast) / is_true_cast.shape[0]
#     return acc_rate_regression

def graph_cl25(dataset):
    ''''''
    dataset = onehot(dataset, 25)
    model_cl25 = Cl25()
    optimizer = tf.keras.optimizers.SGD(lr=1e-2)
    model_cl25.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    train_data, test_data = spliting(dataset, 6000)
    flag = 0
    for epoch in range(10000):
        for train_data_batch in input(dataset=train_data, batch_size=500):
            loss_train, _= model_cl25.train_on_batch(x=[train_data_batch[:, :4], train_data_batch[:, 4:-25]], y=train_data_batch[:, -25:])
            if epoch % 100 == 0 and flag == 0:
                print('第%s轮后训练集损失函数值为: %s' % (epoch, loss_train))
                flag = 1
        if epoch % 100 == 0:
            _, acc = model_cl25.evaluate(x=[test_data[:, :4], test_data[:, 4:-25]], y=test_data[:, -25:], verbose=0)
            print('测试集准确率为: %s' % acc)
        flag = 0


if __name__ == '__main__':
    path = '/home/xiaosong/桌面/pny_cl25.pickle'
    dataset = LoadFile(p=path)
    graph_cl25(dataset=dataset)