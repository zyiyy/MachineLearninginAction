# coding=utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def loadDataSet():
    '''
    加载数据集
    :return: dataMat, labelMat, type=np.array
    '''
    with open('./data/iris/iris.data', 'r') as fr:
        dataMat = []
        labelList = []
        labelDict = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
        for line in fr.readlines():
            lineArr = line.strip().split(',')
            # dataMat.append([1.0, float(lineArr[1]), float(lineArr[3])])
            dataMat.append([1.0, float(lineArr[0]), float(lineArr[1]), float(lineArr[2]), float(lineArr[3])])
            labelList.append(labelDict[lineArr[-1]])
        labelMat = np.zeros((len(labelList), 2))
        for i in range(len(labelList)):
            if labelList[i] == 0:
                labelMat[i][0] = 1
            elif labelList[i] == 1:
                labelMat[i][1] = 1
    return np.array(dataMat), np.array(labelMat), np.array(labelList)


def analyseData(dataMat):
    '''
    分析数据
    :param dataMat:
    :return:
    '''
    data = pd.DataFrame(dataMat, columns=['-', 'sepal length', 'sepal width', 'petal length', 'petal width'])
    print(data.info())
    print(data.head())
    print(data.describe())


def classifyVector(inX, weights):
    '''
    分类函数
    :param inX: 输入样本
    :param weights: 权值
    :return: 类别0, 1, 2
    '''
    inX = np.mat(inX)
    predict = np.exp(inX * weights)
    if predict.max() < 1:
        return 2
    else:
        return np.argmax(predict)



def error(dataMatrix, labelMat, weights):
    m, n = np.shape(labelMat)
    divided = np.ones(m, dtype=float)
    error = np.zeros((m, n), dtype=float)
    for i in range(m):
        for j in range(n):
            # print(weights[:, j])
            # print(dataMatrix[:, i])
            divided[i] = divided[i] + np.exp(weights[:, j].transpose() * dataMatrix[:, i])
    # print(divided[0])
    for i in range(m):
        for j in range(n):
            error[i][j] = np.exp(weights[:, j].transpose() * dataMatrix[:, i]) / divided[i]
    error = labelMat - error
    return error


# 批梯度下降
def batchGradientDescent(dataMatIn, labelMat, weights, maxCycles=200, lambd=0):
    dataMatrix = dataMatIn.transpose()  # n * m
    # 全部变成矩阵运算, 不易出错
    dataMatrix = np.mat(dataMatrix)
    labelMat = np.mat(labelMat)
    for k in range(maxCycles):
        alpha = 0.01
        # 全部变成矩阵运算, 不易出错
        errorMat = error(dataMatrix, labelMat, np.mat(weights))
        # 此处weight变成np.mat, 计算内积时要注意, 尤其是np.sum()
        weights = (1 - alpha * lambd) * weights + alpha * dataMatrix * errorMat
    return weights


def irisTest(weights):
    '''
    在iris数据集上测试效果
    :param weights: 训练之后的权值
    :return:
    '''
    dataMat, labelMat, labelList = loadDataSet()
    errorCount = 0
    for i in range(dataMat.shape[0]):
        predict = classifyVector(dataMat[i], weights)
        print(predict, labelList[i])
        if predict != labelList[i]:
            errorCount += 1
    return errorCount


# 取两个维度可视化
def view():
    dataMat, labelMat, _ = loadDataSet()
    dataMat = np.concatenate((dataMat[:, 0].reshape(-1, 1), dataMat[:, 2].reshape(-1, 1), dataMat[:, 4].reshape(-1, 1)), axis=1)
    # print(dataMat)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(dataMat[:50, 1], dataMat[:50, 2], color='red', marker='o', label='setosa')  # 前50个样本的散点图
    ax.scatter(dataMat[50:100, 1], dataMat[50:100, 2], color='blue', marker='x', label='versicolor')  # 中间50个样本的散点图
    ax.scatter(dataMat[100:, 1], dataMat[100:, 2], color='green', marker='+', label='Virginica')  # 后50个样本的散点图

    weights = batchGradientDescent(dataMat, labelMat, weights=np.ones((3, 2)), lambd=0.01)

    yy1 = np.zeros(40)
    xx1 = np.arange(2, 6, 0.1)  # 定义x的范围，像素为0.1
    for i in range(len(xx1)):
        yy1[i] = (- weights[:, 0][0] - weights[:, 0][1] * xx1[i]) / weights[:, 0][2]
    plt.plot(xx1, yy1, color='yellow')

    yy2 = np.zeros(40)
    xx2 = np.arange(2, 6, 0.1)  # 定义x的范围，像素为0.1
    for i in range(len(xx2)):
        yy2[i] = (- weights[:, 1][0] - weights[:, 1][1] * xx2[i]) / weights[:, 1][2]
    plt.plot(xx2, yy2, color='red')
    plt.show()


if __name__ == '__main__':
    dataMat, labelMat, _ = loadDataSet()
    # print(dataMat.shape)    # 150 * 5
    # print(labelMat.shape)   # 150 * 2
    # analyseData(dataMat)

    view()

    weights = batchGradientDescent(dataMat, labelMat, weights=np.ones((len(dataMat[0]), 2)), lambd=0.01)
    print(weights)
    errorCount = irisTest(weights)
    print(errorCount)

