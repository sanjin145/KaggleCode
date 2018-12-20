#!/usr/bin/env python
# encoding: utf-8

"""
@version: V1.0.0
@author: gold_worker
@contact: dingxin_hz@163.com
@time: 2018-12-11 下午 4:38
"""

import numpy as np
import pandas as pd
import time
import matplotlib as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

def digitRecognizeKnnCrossVal():
    #获取训练数据和测试数据
    pdTrainDataSet = pd.read_csv("../train.csv")
    ndTrainDataSet = pdTrainDataSet.values

    pdTestDataSet = pd.read_csv("../test.csv")
    ndTestDataSet = pdTestDataSet.values

    #分割特征和标签
    trainDataSet = ndTrainDataSet[:,1:]
    trainDataSetLabel = ndTrainDataSet[:,0]

    #创建验证用小数据集
    trainDataSetSmall = trainDataSet[:5000]
    trainDataSetLabelSmall = trainDataSetLabel[:5000]

    #创建KNN分类器,进行交叉验证调参
    neigh = KNeighborsClassifier()
    params = {
        "n_neighbors":range(2,5),
        "algorithm":["kd_tree"]
    }
    gsearch = GridSearchCV(neigh,params,cv=5)
    gsearch.fit(trainDataSetSmall,trainDataSetLabelSmall)
    print(gsearch.cv_results_)
    print(gsearch.best_params_)
    print(gsearch.best_score_)

    #score = cross_val_score(neigh,trainDataSetSmall,trainDataSetLabelSmall,cv=5)
    #print(score.mean())

    #正式训练测试集
    #neigh.fit(trainDataSet,trainDataSetLabel)
    #predResultSet = neigh.predict(ndTestDataSet)
    #print(type(predResultSet)) #ndarray

    #写入训练结果到提交文件
    # index = []
    # for i in range(predResultSet.shape[0]):
    #     index.append(i+1)
    #
    # predDataFrame = pd.DataFrame({"ImageId":index,"Label":predResultSet})
    # predDataFrame.to_csv("brute_knn_submission.csv",index=False)


if __name__ == "__main__":
    startTime = time.clock()
    digitRecognizeKnnCrossVal()
    print("[Finished in %ds]"%(time.clock()-startTime))#23分钟
