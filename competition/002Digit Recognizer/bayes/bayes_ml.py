#!/usr/bin/env python
# encoding: utf-8

"""
@version: V1.0.0
@author: gold_worker
@contact: dingxin_hz@163.com
@time: 2018-12-12 下午 4:07
"""

import numpy as np
import pandas as pd
import time
import matplotlib as plt
from sklearn import naive_bayes
from sklearn.model_selection import cross_val_score

def loadData():
    # 从文件中获取训练数据
    pdTrainData = pd.read_csv("../train.csv")
    ndTrainData = pdTrainData.values

    # 从文件中获取测试数据
    pdTestData = pd.read_csv("../test.csv")
    ndTestData = pdTestData.values

    #将训练数据的特征数据和标签分开
    trainDataSet = ndTrainData[:,1:]
    trainDataSetLabel = ndTrainData[:,0]

    return trainDataSet,trainDataSetLabel,ndTestData

#利用sklearn包，进行test样本集分类判别
def classify(trainDataSet, trainDataSetLabel, testDataSet):
    #创建
    bayes_clf = naive_bayes.MultinomialNB()

    # score = cross_val_score(bayes_clf,trainDataSet,trainDataSetLabel,cv=5)
    # print(score.mean())

    bayes_clf.fit(trainDataSet, trainDataSetLabel)
    pred_result = bayes_clf.predict(testDataSet)

    return pred_result

#将预测结果写入文件中
def writeToFile(predResult):
    index = []
    for i in range(predResult.shape[0]):
        index.append(i+1)
        if i < 5:
            print(index)

    print(predResult.shape)
    print(predResult[:3])
    dataDict = {"ImageId":index,"Label":predResult}
    pdDateFrameData = pd.DataFrame({"ImageId":index,"Label":predResult})

    #没有加index=False,前面会多一列序号列
    pdDateFrameData.to_csv("naive_bayes_submission.csv",index=False)

def navieBayesDigitRecognize():
    trainDataSet,trainDataSetLabel,testDataSet = loadData()
    classify(trainDataSet, trainDataSetLabel, testDataSet)
    predResult = classify(trainDataSet,trainDataSetLabel,testDataSet)
    writeToFile(predResult)

if __name__ == "__main__":
    startTime = time.clock()
    navieBayesDigitRecognize()
    endTime = time.clock()
    print("[Finished in %ds]:"%(endTime-startTime))
