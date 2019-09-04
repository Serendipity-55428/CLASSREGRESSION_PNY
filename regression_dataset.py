#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@contact: 1243049371@qq.com
@software: Pycharm
@file: regression_dataset
@time: 2019/9/4 下午3:06
'''
import numpy as np
import pandas as pd
import pickle
import os
from collections import Counter
from classifier_dataset import LoadFile, SaveFile, classifier_25, checkclassifier, dataset_junheng, guiyi, fft_transformer

def dataset_regression_guiyi(dataset, space):
    '''
    对数据集中所有最优半径数值进行区间内的归一化
    :param dataset: 数据集
    :param space: 区间列表
    :return: 处理后数据集
    '''
    dataset_return = np.array([0])
    dataset_pd = pd.DataFrame(data=dataset, columns=['F' + str(i) for i in range(dataset.shape[-1] - 1)] + ['r'])
    for inf, sup in space:
        dataset_sub_pd = dataset_pd.loc[dataset_pd['r'] >= inf]
        dataset_sub_pd = dataset_sub_pd.loc[dataset_sub_pd['r'] < sup]
        if dataset_sub_pd.values.shape[0] == 0: continue
        #添加半径所在的范围列
        dataset_sub_pd['r_sub'] = [(inf, sup)] * (dataset_sub_pd.values.shape[0])
        dataset_r_sub = np.array(list(dataset_sub_pd['r_sub'].values))
        dataset_r = dataset_sub_pd['r'].values
        print(dataset_r.shape)
        dataset_sub_pd['r'] = (dataset_r - dataset_r_sub[:, 0]) / (dataset_r_sub[:, 1] - dataset_r_sub[:, 0])
        dataset_sub_pd.drop(['r_sub'], axis=1, inplace=True)
        dataset_return = dataset_sub_pd.values if dataset_return.any() == 0 else \
            np.vstack((dataset_return, dataset_sub_pd.values))
    # print(np.min(dataset_return[:, -1]), np.max(dataset_return[:, -1]))
    return dataset_return

if __name__ == '__main__':
    space = classifier_25(26)
    # print(space)
    p = '/home/xiaosong/pny相关数据/data_pny/PNY_all.pickle'
    dataset = LoadFile(p)
    dataset_guiyi_sub = dataset_regression_guiyi(dataset, space)
    dataset_4feature, dataset_dense, label = dataset_guiyi_sub[:, :4], dataset_guiyi_sub[:, 4:-1], \
                                             dataset_guiyi_sub[:, -1][:, np.newaxis]
    dataset_fft = fft_transformer(dataset_dense, 100)
    dataset = np.hstack((dataset_4feature, dataset_fft, label))
    dataset_guiyi = guiyi(dataset)
    print(dataset_guiyi.shape)
    # SaveFile(data=dataset_guiyi, savepickle_p='/home/xiaosong/桌面/pny_cl25.pickle')
    # print(np.max(dataset_guiyi, axis=0))
