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

