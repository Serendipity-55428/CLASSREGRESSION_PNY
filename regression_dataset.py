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

def dataset_regression_guiyi:
    ''''''

if __name__ == '__main__':
    space = classifier_25(26)
    # print(space)
    p = '/home/xiaosong/pny相关数据/data_pny/PNY_all.pickle'
    dataset = LoadFile(p)
