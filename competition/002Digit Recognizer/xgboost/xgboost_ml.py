#!/usr/bin/env python
# encoding: utf-8

"""
@version: V1.0.0
@author: gold_worker
@contact: dingxin_hz@163.com
@time: 2018-12-13 下午 2:21
"""

import numpy as np
import pandas as pd
import time
import matplotlib as plt
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

def loadData():
    # 从文件中获取数据
    pdTrainDataSet = pd.read_csv("../train.csv")
    ndTrainDataSet = pdTrainDataSet.values

    pdTestDataSet = pd.read_csv("../test.csv")
    ndTestDataSet = pdTestDataSet.values

    # 将训练集中特征数据和标签分隔开
    trainDataSet = ndTrainDataSet[:, 1:]
    trainDataSetLabel = ndTrainDataSet[:, 0]

    return trainDataSet, trainDataSetLabel, ndTestDataSet

def classify(trainDataSet, trainDataSetLabel, ndTestDataSet):
    # 创建xgboost分类器
    xgb_clf = XGBClassifier()
    # score = cross_val_score(xgb_clf,trainDataSet,trainDataSetLabel,cv=3)
    # print(score.mean())

    xgb_clf = XGBClassifier().fit(trainDataSet, trainDataSetLabel)
    pred_result = xgb_clf.predict(ndTestDataSet)
    return pred_result

#将预测结果写入文件中
def writeToFile(predResult):
    index = []
    for i in range(predResult.shape[0]):
        index.append(i+1)

    dataDict = {"ImageId":index,"Label":predResult}
    pdDateFrameData = pd.DataFrame(dataDict)

    #没有加index=False,前面会多一列序号列
    pdDateFrameData.to_csv("xgboost_submission.csv",index=False)

def xgboost_handle():
    trainDataSet, trainDataSetLabel, ndTestDataSet = loadData()
    trainSize = 5000
    predResult = classify(trainDataSet[:], trainDataSetLabel[:], ndTestDataSet[:])
    writeToFile(predResult)


if __name__ == "__main__":
    startTime = time.clock()
    xgboost_handle()
    endTime = time.clock()
    print("[Finished in %ds]:" % (endTime - startTime))#cost time = 570s CV = 3

