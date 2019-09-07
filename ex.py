#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@contact: 1243049371@qq.com
@software: Pycharm
@file: ex
@time: 2019/9/2 下午8:23
'''
import numpy as np
import tensorflow as tf
class CNN(tf.keras.Model):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer0 = tf.keras.layers.Reshape(input_shape=(28, 28), target_shape=[28, 28, 1])
        print(self.layer0.get_config())
        self.layer1 = tf.keras.layers.Conv2D(filters=32, kernel_size=[5, 5], padding='same', activation=tf.keras.activations.relu,
                                       kernel_initializer=tf.keras.initializers.TruncatedNormal())
        print(self.layer1.get_config())
        self.layer2 = tf.keras.layers.Conv2D(filters=64, kernel_size=[5, 5], padding='same', activation=tf.nn.relu,
                                       kernel_initializer=tf.keras.initializers.TruncatedNormal)
        # print(self.layer2.get_config())
        self.layer3 = tf.keras.layers.Flatten()
        print(self.layer3.get_config())
        self.layer4 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu, use_bias=True,
                                      kernel_initializer=tf.keras.initializers.TruncatedNormal,
                                      bias_initializer=tf.keras.initializers.TruncatedNormal)
        # print(self.layer4.get_config())
        self.layer5 = tf.keras.layers.Dense(units=10, activation=tf.nn.softmax, use_bias=True,
                                      kernel_initializer=tf.keras.initializers.TruncatedNormal,
                                      bias_initializer=tf.keras.initializers.TruncatedNormal)
        # print(self.layer5.get_config())
    def call(self, inputs, **kwargs):
        cnn = self.layer0(inputs)
        cnn = self.layer1(cnn)
        cnn = self.layer2(cnn)
        cnn = self.layer3(cnn)
        cnn = self.layer4(cnn)
        return self.layer5(cnn)

# input = tf.keras.Input(shape=(28, 28))
# cnn = tf.keras.layers.Reshape(target_shape=[28, 28, 1])(input)
# cnn = tf.keras.layers.Conv2D(filters=32, kernel_size=[5, 5], padding='same', activation=tf.nn.relu,
#                                      kernel_initializer=tf.keras.initializers.TruncatedNormal())(cnn)
# cnn = tf.keras.layers.Conv2D(filters=64, kernel_size=[5, 5], padding='same', activation=tf.nn.relu,
#                                      kernel_initializer=tf.keras.initializers.TruncatedNormal())(cnn)
# cnn = tf.keras.layers.Flatten()(cnn)
# cnn = tf.keras.layers.Dense(units=100, activation=tf.nn.relu, use_bias=True,
#                             kernel_initializer=tf.keras.initializers.TruncatedNormal(),
#                             bias_initializer=tf.keras.initializers.TruncatedNormal())(cnn)
# cnn = tf.keras.layers.Dense(units=10, activation=tf.nn.softmax, use_bias=True,
#                             kernel_initializer=tf.keras.initializers.TruncatedNormal(),
#                             bias_initializer=tf.keras.initializers.TruncatedNormal())(cnn)
# cnn = tf.keras.Model(inputs=input, outputs=cnn)

def train(x_train, y_train, x_test, y_test):
    cnn = CNN()
    optimizer = tf.keras.optimizers.Adam(lr=1e-2)
    cnn.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    for i in range(1):
        for j in range(0, 10000, 500):
            xt, yt = x_train[j:j+500], y_train[j:j+500]
            a = cnn.train_on_batch(x=xt, y=yt)
            print(a)
        cnn.evaluate(x=x_test[:3000], y=y_test[:3000], verbose=0)
        c = cnn.predict(x=x_test[:3000])
    # print(cnn.get_config())
    cnn.save(filepath='/home/xiaosong/桌面/ex.h5')


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255, x_test / 255
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    train(x_train, y_train, x_test, y_test)
    # model = tf.keras.models.load_model('/home/xiaosong/桌面/ex.h5')
    # model.evaluate(x=x_test[:3000], y=y_test[:3000], verbose=1)
    # a = tf.keras.layers.Conv2D(filters=32, kernel_size=[5, 5])
    # print(a.get_config())
    # import numpy  # 导入数据库
    # from keras.datasets import mnist
    # from keras.models import Sequential
    # from keras.layers import Dense
    # from keras.layers import Dropout
    # from keras.utils import np_utils
    #
    # seed = 7  # 设置随机种子
    # numpy.random.seed(seed)
    #
    # (X_train, y_train), (X_test, y_test) = mnist.load_data()  # 加载数据
    #
    # num_pixels = X_train.shape[1] * X_train.shape[2]
    # X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
    # X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')
    # # 数据集是3维的向量（instance length,width,height).对于多层感知机，模型的输入是二维的向量，因此这
    # # 里需要将数据集reshape，即将28*28的向量转成784长度的数组。可以用numpy的reshape函数轻松实现这个过
    # # 程。
    #
    # # 给定的像素的灰度值在0-255，为了使模型的训练效果更好，通常将数值归一化映射到0-1。
    # X_train = X_train / 255
    # X_test = X_test / 255
    #
    # # 最后，模型的输出是对每个类别的打分预测，对于分类结果从0-9的每个类别都有一个预测分值，表示将模型
    # # 输入预测为该类的概率大小，概率越大可信度越高。由于原始的数据标签是0-9的整数值，通常将其表示成#0ne-hot向量。如第一个训练数据的标签为5，one-hot表示为[0,0,0,0,0,1,0,0,0,0]。
    #
    # y_train = np_utils.to_categorical(y_train)
    # y_test = np_utils.to_categorical(y_test)
    # num_classes = y_test.shape[1]
    #
    #
    # # 现在需要做得就是搭建神经网络模型了，创建一个函数，建立含有一个隐层的神经网络。
    # # define baseline model
    # def baseline_model():
    #     # create model
    #     model = Sequential()
    #     model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    #     model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    #     # Compile model
    #     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #     return model
    #
    #
    # # 型的隐含层含有784个节点，接受的输入长度也是784（28*28），最后用softmax函数将预测结果转换为标签
    # # 的概率值。
    # # 将训练数据fit到模型，设置了迭代轮数，每轮200个训练样本，将测试集作为验证集，并查看训练的效果。
    #
    # # build the model
    # model = baseline_model()
    # # Fit the model
    # model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
    # # Final evaluation of the model
    # scores = model.evaluate(X_test, y_test, verbose=0)
    #
