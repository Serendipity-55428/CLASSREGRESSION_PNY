#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@contact: 1243049371@qq.com
@software: Pycharm
@file: Predict
@time: 2019/9/6 下午4:34
'''
import tensorflow as tf
import numpy as np
import os
import pandas as pd
from classifier_dataset import classifier_25, SaveFile, LoadFile, fft_transformer, guiyi
from  re_sub import acc_regression

if __name__ == '__main__':
    path = '/home/xiaosong/pny相关数据/data_pny/PNY_all.pickle'
    path_cl = '/home/xiaosong/桌面/graph_cl_re/graph_cl.h5'
    path_re = '/home/xiaosong/桌面/graph_cl_re/graph_re.h5'
    model_cl = tf.keras.models.load_model(filepath=path_cl)
    model_re = tf.keras.models.load_model(filepath=path_re)
    space_list = classifier_25(n=26)
    dataset = LoadFile(p=path)
    dataset_4feature, dataset_dense, label = dataset[:, :4], dataset[:, 4:-1], \
                                             dataset[:, -1][:, np.newaxis]
    dataset_fft = fft_transformer(dataset_dense, 100)
    dataset = np.hstack((dataset_4feature, dataset_fft, label))
    dataset_guiyi = guiyi(dataset)
    print(dataset_guiyi.shape)
    rng = np.random.RandomState(0)
    rng.shuffle(dataset_guiyi)
    test_data = dataset_guiyi[:6000, :] #根据导入数据标签改
    print(test_data.shape)
    result_cl = model_cl.predict(x=[test_data[:, :4], test_data[:, 4:-1]], verbose=0)
    result_cl = np.argmax(a=result_cl, axis=1)
    result_inf = result_cl * 10 #将分类器执行后的结果对应到各个子分类空间中
    #regression
    result_re = model_re.predict(x=[test_data[:, :4], test_data[:, 4:-1]], verbose=0)
    re_guiyi_len = lambda x: (space_list[x][-1] - space_list[x][0]) #类别区间长度向量函数
    re_guiyi_inf = lambda x: space_list[x][0] #类别区间左端点向量函数
    func_len = np.frompyfunc(re_guiyi_len, 1, 1)
    func_inf = np.frompyfunc(re_guiyi_inf, 1, 1)
    result_re_len = func_len(result_cl)[:, np.newaxis] #类别区间长度
    print(result_re_len.shape)
    result_re_inf = func_inf(result_cl)[:, np.newaxis] #类别区间左端点
    print(result_re_inf.shape)
    r_fin = result_re_inf + result_re * result_re_len #测试集最终预测半径
    print(result_re.shape)
    #计算经过分类和回归器后的准确率
    T_list = [7, 6, 5, 3, 2, 1, 0.5]
    print(test_data[:, -1].shape, r_fin.shape)
    acc_list = (acc_regression(y_true=test_data[:, -1][:, np.newaxis], y_pred=r_fin, Threshold=t) for t in T_list)
    print('整个回归模型的准确率与阈值选择对应关系为:')
    for t, acc in zip(T_list, acc_list):
        print('T:%s Acc:%s' % (t, acc))

