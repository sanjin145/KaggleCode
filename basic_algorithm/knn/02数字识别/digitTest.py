#!/usr/bin/env python
# encoding: utf-8

"""
@version: V1.0.0
@author: gold_worker
@contact: dingxin_hz@163.com
@time: 2018-12-10 下午 9:11
"""

import numpy as np
import time
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as kNN

"""
函数说明:将32x32的二进制图像转换为1x1024向量。

Parameters:
    filename - 文件名
Returns:
    returnVect - 返回的二进制图像的1x1024向量

Modify:
    2018-12-10
"""
def img2vector(filename):
    # 创建1 * 1024零向量
    returnVect = np.zeros((1, 1024))
    # 打开文件
    fr = open(filename)
    # 按行读取
    for i in range(32):
        # 读一行数据
        lineStr = fr.readline()
        # 每一行的前32个元素依次添加到returnVect中
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    # 返回转换后的1 * 1024向量
    return returnVect

"""
函数说明:手写数字分类测试

Parameters:
    无
Returns:
    无

Modify:
    2018-12-10
"""
def handwritingClassTest():
    # 测试集的Labels
    hwLabels = []
    # 返回trainingDigits目录下的文件名
    absPath = "E:\\Code\\KaggleCode\\basic_algorithm\\knn\\02数字识别\\"
    trainingFileList = listdir(absPath + "trainingDigits")
    # 返回训练文件夹下文件的个数
    trainDataSize = len(trainingFileList)
    # 初始化训练的Mat矩阵，测试集
    trainingMat = np.zeros((trainDataSize, 1024))
    # 从文件名中解析出训练样本的类别
    for i in range(trainDataSize):
        # 获得文件的名称
        fileNameStr = trainingFileList[i]
        # 获得分类的数字
        classNumber = int(fileNameStr.split("_")[0])
        # 将获得的类别添加到hwLabels数组中
        hwLabels.append(classNumber)
        # 将每一个文件的1*1024数据存储到trainingMat矩阵中
        trainingMat[i, :] = img2vector(absPath + "trainingDigits/%s" % (fileNameStr))

    # 构建kNN分类器
    neigh = kNN(n_neighbors=3, algorithm='brute')
    # 拟合模型，trainingMat为训练矩阵，hwLabels为对应的标签
    neigh.fit(trainingMat, hwLabels)

    # 返回testDigit目录下的文件列表
    testFileList = listdir(absPath + "testDigits")
    # 错误检测计数
    errorCount = 0.0
    # 测试数据的数量
    testDataSize = len(testFileList)

    # 从文件中解析出测试集的类别并进行分类测试
    for i in range(testDataSize):
        # 获取测试文件的名称
        testFileNameStr = testFileList[i]
        # 获取测试集正确的分类数字
        rightClassNumber = int(testFileNameStr.split("_")[0])
        # 获得测试集的 1 * 1024向量，用于训练
        vectorUnderTest = img2vector(absPath + 'testDigits\\%s' % (testFileNameStr))
        # 获得预测结果
        classifierResult = neigh.predict(vectorUnderTest)
        print("分类返回结果为%d,真实结果为%d" % (classifierResult, rightClassNumber))
        if classifierResult != rightClassNumber:
            errorCount += 1.0
    print("总共错了%d个数据\n,错误率为%f%%" % (errorCount, errorCount / testDataSize * 100))

if __name__ == "__main__":
    start = time.clock()
    handwritingClassTest()
    end = time.clock()
    print("[Finished in %ds]" % (end - start))