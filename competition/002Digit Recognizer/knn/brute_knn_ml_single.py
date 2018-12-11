#!/usr/bin/env python
# encoding: utf-8

"""
@description: kaggle数字识别线性扫描分类
@version: V1.0.0
@author: gold_worker
@contact: dingxin_hz@163.com
@time: 2018-12-11 下午 12:55
"""

import numpy as np
import pandas as pd
import time
from sklearn.neighbors import KNeighborsClassifier

"""
函数说明:Kaggle线性扫描KNN数字分类

Parameters:
    无
Returns:
    无

Modify:
    2018-12-11
"""
def digitRecognizer():
    # 读取训练集数据
    pdTrainData = pd.read_csv("../train.csv")
    # print(pdTrainData.head())  # [5 rows x 785 columns]
    #print(type(pdTrainData))  # <class 'pandas.core.frame.DataFrame'>
    # 转换成ndarray数组对象
    trainSet = pdTrainData.values
    # print(type(trainSet))
    # 对ndarray对象进行切片操作，分隔开标签和特征列
    trainSetLabels = trainSet[:, 0]
    trainSet = trainSet[:, 1:]

    # 构造测试集数据
    pdTestData = pd.read_csv("../test.csv")
    print(pdTestData.head())
    # values方法会移除DataFrame的轴标签，返回一个ndarray类型的数据
    testSet = pdTestData.values


    # 保存结果的列表
    index = []
    result = []
    # 对测试集中的每条数据进行处理
    for i in range(5):
        print("开始判断第%d条数据" % i)
        index.append(i+1)
        predictNum = classify(testSet[i],trainSet,trainSetLabels,3)
        result.append(predictNum[0])

    #将数据保存到csv文件中
    predictions = pd.DataFrame({"ImageId":index,"Label":result})
    predictions.to_csv("brute_knn_submission.csv",index=False)

def classify(testData,trainSet,trainSetLabels,knum):
    #print(testData)
    neigh = KNeighborsClassifier(n_neighbors=3,algorithm='kd_tree')
    neigh.fit(trainSet,trainSetLabels)
    #predict方法必须是一个二维矩阵，哪怕只有一条数据
    y_pred = neigh.predict(testData.reshape(1,-1))
    return y_pred

if __name__ == "__main__":
    startTime = time.clock()
    digitRecognizer()
    endTime = time.clock()
    print("[Finished in %ds]" % (endTime - startTime))
