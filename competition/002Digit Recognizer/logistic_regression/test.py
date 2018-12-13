#!/usr/bin/env python
# encoding: utf-8

"""
@version: V1.0.0
@author: gold_worker
@contact: dingxin_hz@163.com
@time: 2018-12-12 下午 7:24
"""

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    # 直接返回sigmoid函数
    return (1. + np.exp(-x))

def plot_sigmoid():
    # param:起点，终点，间距
    x = np.arange(-8, 8, 0.2)
    y = sigmoid(x)

    fig, ax = plt.subplots(1, 1)
    age = sigmoid(-3.8)
    print(age)
    ax.set_title("age=%d"%(int(age)))
    ax.plot(x,y)

    plt.show()

if __name__ == '__main__':
    plot_sigmoid()

