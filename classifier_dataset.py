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
##################最优半径区间划分为25个子区间###################
def classifier_25():
    ''''''
    space = np.linspace(0, 250, 26)
    space_sub1 = space[0:-1]
    space_sub2 = space[1:]
    space_sub2[-1] = 252
    space_sub1 = list(space_sub1)
    space_sub2 = list(space_sub2)
    space = zip(space_sub1, space_sub2)
    print(list(space))
    return list(space)


if __name__ == '__main__':
    classifier_25()
