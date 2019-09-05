#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@contact: 1243049371@qq.com
@software: Pycharm
@file: cl25
@time: 2019/9/4 下午10:17
'''
import tensorflow as tf
import numpy as np
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
        layer = tf.concat(values=[flat, input1], axis=1)
        layer = tf.keras.layers.Dense(units=100, activation=tf.nn.relu, use_bias=True,
                                      kernel_initializer=tf.keras.initializers.TruncatedNormal,
                                      bias_initializer=tf.keras.initializers.TruncatedNormal, name='x_fc1')(layer)
        layer = tf.keras.layers.Dense(units=200, activation=tf.nn.relu, use_bias=True,
                                      kernel_initializer=tf.keras.initializers.TruncatedNormal,
                                      bias_initializer=tf.keras.initializers.TruncatedNormal, name='x_fc2')(layer)
        layer = tf.keras.layers.Dense(units=3, activation=tf.nn.softmax, use_bias=True,
                                       kernel_initializer=tf.keras.initializers.TruncatedNormal,
                                       bias_initializer=tf.keras.initializers.TruncatedNormal, name='output')(layer)

    model_cl25 = tf.keras.Model(inputs=[input1, input2], outputs=layer)
    return model_cl25

def graph_cl25():
    ''''''
    model_cl25 = Cl25()
    optimizer = tf.keras.optimizers.SGD(lr=1e-2)
    model_cl25.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    for epoch in range(10000):



if __name__ == '__main__':
    pass