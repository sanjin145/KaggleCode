#!/usr/bin/env python
# encoding: utf-8

"""
@version: V1.0.0
@author: gold_worker
@contact: dingxin_hz@163.com
@time: 2018-12-12 下午 7:05
"""

import numpy as np
import pandas as pd
import time
import matplotlib as plt
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

#获取数据集
def loadData():
    #从文件中获取数据
    pdTrainDataSet = pd.read_csv("../train.csv")
    ndTrainDataSet = pdTrainDataSet.values

    pdTestDataSet = pd.read_csv("../test.csv")
    ndTestDataSet = pdTestDataSet.values

    #将训练集中特征数据和标签分隔开
    trainDataSet = ndTrainDataSet[:,1:]
    trainDataSetLabel = ndTrainDataSet[:,0]

    return trainDataSet,trainDataSetLabel,ndTestDataSet



def classify(trainDataSet,trainDataSetLabel,testDataSet):
    # 创建逻辑回归分类器
    logis_clf = LogisticRegression()
    # score = cross_val_score(logis_clf,trainDataSet,trainDataSetLabel,cv=5)
    # print(score.mean())
    logis_clf.fit(trainDataSet,trainDataSetLabel)
    pred_result = logis_clf.predict(testDataSet)
    return pred_result

#将预测结果写入文件中
def writeToFile(predResult):
    index = []
    for i in range(predResult.shape[0]):
        index.append(i+1)

    dataDict = {"ImageId":index,"Label":predResult}
    pdDateFrameData = pd.DataFrame(dataDict)

    #没有加index=False,前面会多一列序号列
    pdDateFrameData.to_csv("logistic_regression_submission.csv",index=False)

def logistic_regression_handle():
    trainDataSet, trainDataSetLabel, ndTestDataSet = loadData()
    predResult = classify(trainDataSet, trainDataSetLabel, ndTestDataSet)
    writeToFile(predResult)

if __name__ == "__main__":
    startTime = time.clock()
    logistic_regression_handle()
    endTime = time.clock()
    print("[Finished in %ds]:" % (endTime - startTime))#2068s 半个小时

