#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@contact: 1243049371@qq.com
@software: Pycharm
@file: classifier_dataset
@time: 2019/9/3 下午9:43
'''
import numpy as np
import pandas as pd
import pickle
import os
from collections import Counter
##################最优半径区间划分为25个子区间###################

def LoadFile(p):
    '''
    读取文件
    :param p: 数据集绝对路径
    :return: 数据集
    '''
    data = np.array([0])
    try:
        with open(p, 'rb') as file:
            data = pickle.load(file)
    except:
        print('文件不存在!')
    finally:
        return data

def SaveFile(data, savepickle_p):
        '''
        存储整理好的数据
        :param data: 待存储数据
        :param savepickle_p: pickle后缀文件存储绝对路径
        :return: None
        '''
        if not os.path.exists(savepickle_p):
            with open(savepickle_p, 'wb') as file:
                pickle.dump(data, file)

def classifier_25(n):
    '''
    细化最优半径区间
    :param n: 区间个数
    :return: 区间划分列表
    '''
    space = np.linspace(0, 250, n)
    space_sub1 = space[0:-1]
    space_sub2 = space[1:]
    space_sub2[-1] = 252
    space_sub1 = list(space_sub1)
    space_sub2 = list(space_sub2)
    space = zip(space_sub1, space_sub2)
    # print(list(space))
    return list(space)

def dataset_cl(dataset, space):
    ''''''
    dataset_return = np.array([0])
    dataset_pd = pd.DataFrame(data=dataset, columns=['F' + str(i) for i in range(dataset.shape[-1]-1)] + ['r'])
    for inf, sup in space:
        dataset_sub_pd = dataset_pd.loc[dataset_pd['r'] >= inf]
        dataset_sub_pd = dataset_sub_pd.loc[dataset_sub_pd['r'] < sup]
        dataset_sub_pd['r'] = space.index((inf, sup))
        dataset_return = dataset_sub_pd.values if dataset_return.any() == 0 else \
            np.vstack((dataset_return, dataset_sub_pd.values))
    return dataset_return


if __name__ == '__main__':
    space = classifier_25(26)
    p = '/home/xiaosong/pny相关数据/data_pny/PNY_all.pickle'
    dataset = LoadFile(p)
    # print(dataset.shape)
    dataset_cl25 = dataset_cl(dataset=dataset, space=space)
    print(dataset_cl25.shape, Counter(dataset_cl25[:, -1]))