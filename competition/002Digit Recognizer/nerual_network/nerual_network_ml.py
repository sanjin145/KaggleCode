#!/usr/bin/env python
# encoding: utf-8

"""
@version: V1.0.0
@author: gold_worker
@contact: dingxin_hz@163.com
@time: 2018-12-16 下午 6:56
"""

import numpy as np
import pandas as pd
import time
import matplotlib as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

def loadData():
    #从文件中获取数据
    pdTrainData = pd.read_csv("../train.csv")
    pdTestData = pd.read_csv("../test.csv")

    ndTrainData = pdTrainData.values
    ndTestData = pdTestData.values

    #切分训练集标签和特征字段数据
    trainDataSet = ndTrainData[:,1:]
    trainDataSetLabel = ndTrainData[:,0]

    return trainDataSet,trainDataSetLabel,ndTestData

def classify(trainDataSet,trainDataSetLabel,testDataSet):
    #nerual_network_clf = MLPClassifier(solver='sgd', activation='relu',alpha=1e-5,
                        #hidden_layer_sizes=(50, 50), random_state=1)
    nerual_network_clf = MLPClassifier()
    param_test = {
        'hidden_layer_sizes': [(i*20,)  for i in range(5,10)],
        'activation':['relu'],
        'solver':['sgd'],
        'max_iter':[500]
    }
    gsearch = GridSearchCV(nerual_network_clf, param_grid=param_test,  cv=5)
    gsearch.fit(trainDataSet,trainDataSetLabel)
    print(gsearch.cv_results_)
    print(gsearch.best_params_)
    print(gsearch.best_score_)

    # 训练模型
    #nerual_network_clf.fit(trainDataSet,trainDataSetLabel)
    #pred_result = nerual_network_clf.predict(testDataSet)

    #return pred_result

def writeToFile(predResult):
    index = []
    for i in range(predResult.shape[0]):
        index.append(i + 1)

    dataDict = {"ImageId": index, "Label": predResult}
    pdDateFrameData = pd.DataFrame(dataDict)

    # 没有加index=False,前面会多一列序号列
    pdDateFrameData.to_csv("nerual_network_submission.csv", index=False)

def neural_network_handle():
    trainDataSet,trainDataSetLabel,testDataSet = loadData()
    predictResult = classify(trainDataSet[:],trainDataSetLabel[:],testDataSet)
    #writeToFile(predictResult)

if __name__ == "__main__":
    startTime = time.clock()
    neural_network_handle()
    endTime = time.clock()
    print("[Finished in %ds]:" % (endTime - startTime))

