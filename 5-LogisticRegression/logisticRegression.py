# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt


def loadDataSet():
    dataMat = []; labelMat = []
    with open('./data/testSet.txt', 'r') as fr:
        for line in fr.readlines():
            lineArr = line.strip().split()
            dataMat.append([1.0, float(lineArr[0]) ,float(lineArr[1])])
            labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inX):
    return 1 / (1 + np.exp(-inX))


# 批量梯度下降
def batchGradientDescent(dataMatIn, classLabels, weights, maxCycles=500):
    weightsList = []
    dataMatrix = np.mat(dataMatIn).transpose()
    labelMat = np.mat(classLabels).transpose()
    alpha = 0.01
    for k in range(maxCycles):
        # alpha = 4 / (1.0 + k) + 0.01
        h = sigmoid(weights.transpose() * dataMatrix)
        error = labelMat - h.transpose()
        weights += np.array(alpha * dataMatrix * error).squeeze()
        weightsList.append([k, weights[0], weights[1], weights[2]])
        # print(loss(dataMatIn, classLabels, weights))
    return weights, weightsList


# 随机梯度下降, 优化后
def stochasticGradientDescent(dataMatIn, classLabels, weights, maxCycles=500):
    weightsList = []
    dataMatrix = np.mat(dataMatIn).transpose()
    labelMat = np.mat(classLabels).transpose()
    # alpha = 0.01
    for k in range(maxCycles):
        dataIndex = range(dataMatrix.shape[1])
        for i in range(dataMatrix.shape[1]):
            alpha = 4 / (1.0 + k + i) + 0.0001    # 加快收敛, alpha非严格下降
            randIndex = int(np.random.uniform(0, len(dataIndex)))   # 随机选取样本, 减少周期性的波动
            h = sigmoid(weights.transpose() * dataMatrix[:, randIndex])
            error = labelMat[randIndex] - h
            weights += np.array(alpha * error[0, 0] * dataMatrix[:, randIndex]).squeeze()
            weightsList.append([k, weights[0], weights[1], weights[2]])
            del(dataIndex[randIndex])
            # print(loss(dataMatIn, classLabels, weights))
    return weights, weightsList


# 随机梯度下降
def stochasticGradientDescent0(dataMatIn, classLabels, weights, maxCycles=500):
    weightsList = []
    dataMatrix = np.mat(dataMatIn).transpose()
    labelMat = np.mat(classLabels).transpose()
    alpha = 0.01
    for k in range(maxCycles):
        for i in range(dataMatrix.shape[1]):
            h = sigmoid(weights.transpose() * dataMatrix[:, i])
            error = labelMat[i] - h
            weights += np.array(alpha * error[0, 0] * dataMatrix[:, i]).squeeze()
            weightsList.append([k, weights[0], weights[1], weights[2]])
            # print(loss(dataMatIn, classLabels, weights))
    return weights, weightsList


# 返回当前损失函数的值
def loss(dataMatIn, classLabels, weights):
    res = 0.0
    dataMatrix = np.mat(dataMatIn).transpose()
    labelMat = np.mat(classLabels).transpose()
    for i in range(dataMatrix.shape[1]):
        res += labelMat[i] * weights.transpose() * dataMatrix[:, i] - np.log(1 + np.exp(weights.transpose() * dataMatrix[:, i]))
    return -res


def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(len(dataMat)):
        if labelMat[i] == 1:
            xcord1.append(dataMat[i][1])
            ycord1.append(dataMat[i][2])
        else:
            xcord2.append(dataMat[i][1])
            ycord2.append(dataMat[i][2])
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(xcord1, ycord1, c='g', label='class1', marker='s')
    ax.scatter(xcord2, ycord2, c='r', label='class0')
    ax.legend(loc='upper left')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (- weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    plt.show()


# 可视化权值向量的收敛情况
def plotWeightsList(weightsList):
    fig = plt.figure()
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)
    ax1.plot(weightsList[:, 0], weightsList[:, 1])
    ax2.plot(weightsList[:, 0], weightsList[:, 2])
    ax3.plot(weightsList[:, 0], weightsList[:, 3])
    plt.show()


def classifyVector(inX, weights):
    prob = sigmoid(np.sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colicTest():
    frTrain = open('./data/horseColicTraining.txt')
    frTest = open('./data/horseColicTest.txt')
    # frTest = open('./data/horseColicTraining.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    # trainingWeights, weightsList = batchGradientDescent(trainingSet, trainingLabels, np.ones(len(trainingSet[0])), maxCycles=100000)
    trainingWeights, weightsList = stochasticGradientDescent(trainingSet, trainingLabels, np.ones(len(trainingSet[0])), maxCycles=1000)
    plotWeightsList(np.array(weightsList))
    errorCount = 0.0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingWeights = trainingWeights.squeeze()
        if int(classifyVector(np.array(lineArr), trainingWeights)) != int(float(currLine[21])):
            errorCount += 1.0
    errorRate = float(errorCount) / numTestVec
    print('the error rate of this test is : {}'.format(errorRate))
    return errorRate


def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print('after {} iterations the average error rate is: {}'.format(numTests, errorSum / 10))


if __name__ == '__main__':
    # dataMat, labelMat = loadDataSet()
    #
    # weights, weightsList = batchGradientDescent(dataMat, labelMat, np.ones(len(dataMat[0])))
    # plotBestFit(weights)
    # plotWeightsList(np.array(weightsList))
    #
    # weights, weightsList = stochasticGradientDescent0(dataMat, labelMat, np.ones(len(dataMat[0])))
    # plotBestFit(weights)
    # plotWeightsList(np.array(weightsList))
    #
    # weights, weightsList = stochasticGradientDescent(dataMat, labelMat, np.ones(len(dataMat[0])), maxCycles=50)
    # plotBestFit(weights)
    # plotWeightsList(np.array(weightsList))

    multiTest()