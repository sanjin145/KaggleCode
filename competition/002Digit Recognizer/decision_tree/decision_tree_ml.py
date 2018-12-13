#!/usr/bin/env python
# encoding: utf-8

"""
@version: V1.0.0
@author: gold_worker
@contact: dingxin_hz@163.com
@time: 2018-12-13 上午 09:53
"""

import numpy as np
import pandas as pd
import time
import matplotlib as plt
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

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
    decision_tree_clf = DecisionTreeClassifier(random_state=0)
    #score = cross_val_score(decision_tree_clf,trainDataSet[:],trainDataSetLabel[:],cv=10)
    #print(score.mean())
    # CV=3 cost_time=30s |  CV=5 cost_time=60s  | CV=10 cost_time=140s

    decision_tree_clf.fit(trainDataSet,trainDataSetLabel)
    pred_result = decision_tree_clf.predict(testDataSet)
    return pred_result

#将预测结果写入文件中
def writeToFile(predResult):
    index = []
    for i in range(predResult.shape[0]):
        index.append(i+1)

    dataDict = {"ImageId":index,"Label":predResult}
    pdDateFrameData = pd.DataFrame(dataDict)

    #没有加index=False,前面会多一列序号列
    pdDateFrameData.to_csv("decision_tree_submission.csv",index=False)

def decision_tree_handle():
    trainDataSet, trainDataSetLabel, ndTestDataSet = loadData()
    predResult = classify(trainDataSet, trainDataSetLabel, ndTestDataSet)
    writeToFile(predResult)

if __name__ == "__main__":
    startTime = time.clock()
    decision_tree_handle()
    endTime = time.clock()
    print("[Finished in %ds]:" % (endTime - startTime))#16s

