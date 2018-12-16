#!/usr/bin/env python
# encoding: utf-8

"""
@version: V1.0.0
@author: gold_worker
@contact: dingxin_hz@163.com
@time: 2018-12-12 下午 22:05
"""

import numpy as np
import pandas as pd
import time
import matplotlib as plt
from sklearn.model_selection import cross_val_score
from sklearn import svm

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
    # 创建SVM分类器
    svm_clf = svm.SVC(kernel='rbf',gamma=0.001,C=100.0)
    # score = cross_val_score(svm_clf,trainDataSet,trainDataSetLabel,cv=3)
    # print(score.mean())

    svm_clf.fit(trainDataSet,trainDataSetLabel)
    pred_result = svm_clf.predict(testDataSet)
    return pred_result

#将预测结果写入文件中
def writeToFile(predResult):
    index = []
    for i in range(predResult.shape[0]):
        index.append(i+1)

    dataDict = {"ImageId":index,"Label":predResult}
    pdDateFrameData = pd.DataFrame(dataDict)

    #没有加index=False,前面会多一列序号列
    pdDateFrameData.to_csv("svm_submission.csv",index=False)

def svm_handle():
    trainDataSet, trainDataSetLabel, ndTestDataSet = loadData()
    trainSize = 5000
    predResult = classify(trainDataSet[:trainSize], trainDataSetLabel[:trainSize], ndTestDataSet[:trainSize])
    writeToFile(predResult)

if __name__ == "__main__":
    startTime = time.clock()
    svm_handle()
    endTime = time.clock()
    print("[Finished in %ds]:" % (endTime - startTime))#5814ss 1个半个小时

