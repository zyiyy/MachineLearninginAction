# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt


# 加载简易数据集
def loadSimpData():
    dataMat = np.matrix(
        [
            [1., 2.1],
            [1.5, 1.6],
            [1.3, 1.],
            [1., 1.],
            [2., 1.]
        ]
    )
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels


# 简易数据集可视化
def plotSimpData():
    dataMat, classLabels = loadSimpData()
    xcord0 = []
    ycord0 = []
    xcord1 = []
    ycord1 = []
    for i in range(len(classLabels)):
        if classLabels[i] == 1.0:
            xcord1.append(dataMat[i, 0])
            ycord1.append(dataMat[i, 1])
        else:
            xcord0.append(dataMat[i, 0])
            ycord0.append(dataMat[i, 1])
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(xcord0, ycord0, marker='s', s=90)
    ax.scatter(xcord1, ycord1, marker='o', s=50, c='r')
    plt.title('Decision Stump Test Data')
    plt.show()


def stumpClassifiy(dataMatrix, dimen, threshVal, threshIneq):
    '''
    单层决策树分类函数
    :param dataMatrix: 输入数据集
    :param dimen: 切分维度
    :param threshVal: 阈值
    :param threshIneq: 控制分类方向
    :return: 分类结果
    '''
    retArray = np.ones((dataMatrix.shape[0], 1))
    if threshIneq == 'lt':
        # -1类在阈值左边, 数组过滤实现
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray


def buildStump(dataArr, classLabels, D):
    '''
    遍历stumpClassifiy()函数所有可能输入, 找到基于权重向量的最优单层决策树
    :param dataArr: 数据集
    :param classLabels: 标签
    :param D: 权重向量
    :return: 最优单层决策树
    '''
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m, n = dataMatrix.shape
    numSteps = 10.0
    bestStump = {}
    bestClasEst = np.mat(np.zeros((m, 1)))
    minError = np.inf
    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        # print(rangeMin, rangeMax)
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassifiy(dataMatrix, i, threshVal, inequal)
                errArr = np.mat(np.ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr
                # print('split: dim %d, thresh %.2f, thresh ineqal: %s, weighted error: %.3f' % (i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst


def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    '''
    基于单层决策树的adaBoost训练, 返回单层决策树数组
    :param dataArr:
    :param classLabels:
    :param numIt: 弱分类器数量
    :return:
    '''
    weakClassArr = []
    m = dataArr.shape[0]
    # 初始化权重向量
    D = np.mat(np.ones((m, 1)) / m)
    aggClassEst = np.mat(np.zeros((m, 1)))
    # 产生numIt个单层决策树
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        # print('D: ', D.T)
        # 弱分类器权重, 带权错误率越低, alpha越大
        alpha = float(0.5 * np.log((1 - error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        # print('classEst: ', classEst.T)
        # 样本权重向量分母, 为使权重向量满足概率分布(加起来等于1)
        # print('classLables: ', classLabels)
        # print('alpha: ', alpha)
        # print(- alpha * np.array(classLabels * classEst.squeeze()))
        expon = np.exp(- alpha * np.array(classLabels * classEst.squeeze()))
        expon = expon.reshape(-1, 1)
        # print(expon)
        D = np.multiply(D, expon)   # 新权值向量分子部分
        # print(D)
        D = D / D.sum() # 新权值向量
        # print(D)
        aggClassEst += alpha * classEst
        # print('aggClassEst: ', aggClassEst.T)
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print('total errorRate: ', errorRate)
        if errorRate == 0.0:
            break
    return weakClassArr, aggClassEst


def adaClassify(datToClass, classifierArr):
    '''
    使用AdaBoost进行分类
    :param datToClass:
    :param classifierArr:
    :return: 加权分类结果
    '''
    dataMatrix = np.mat(datToClass)
    m = dataMatrix.shape[0]
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassifiy(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        # print(i, aggClassEst)
    return np.sign(aggClassEst)


# 加载horse数据集
def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    with open(fileName, 'r') as fr:
        for line in fr.readlines():
            lineArr = []
            curLine = line.strip().split('\t')
            for i in range(len(curLine) - 1):
                lineArr.append(float(curLine[i]))
            dataMat.append(lineArr)
            labelMat.append(float(curLine[-1]))
    return np.array(dataMat), np.array(labelMat)


# 在horse数据集上测试AdaBoost性能
def testHorseData():
    datArr, labelArr = loadDataSet('./data/horseColicTraining2.txt')
    classifierArray, aggClassEst = adaBoostTrainDS(datArr, labelArr, numIt=60)
    # aggClassEst是加权预测值, 元素值越大, 代表是正类的概率越大
    plotROC(aggClassEst.T, labelArr)

    testArr, testLabelArr = loadDataSet('./data/horseColicTest2.txt')
    prediction60 = adaClassify(testArr, classifierArray)
    errArr = np.mat(np.ones((testArr.shape[0], 1)))
    errorCount = errArr[prediction60 != np.mat(testLabelArr).T].sum()
    errorRate =  errorCount / testArr.shape[0]
    print(errorCount, testArr.shape[0])
    print('errorRate in test dataSet: ', errorRate)


# 绘制ROC曲线
def plotROC(predStrengths, classLabels):
    cur = (1.0, 1.0) # cursor
    ySum = 0.0  # variable to calculate AUC
    # 正类样本的数量
    numPosClas = sum(np.array(classLabels) == 1.0)
    # y轴步长
    yStep = 1 / float(numPosClas)
    # x轴步长
    xStep = 1 / float(len(classLabels) - numPosClas)
    sortedIndicies = predStrengths.argsort() # get sorted index, it's reverse
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(1, 1, 1)
    # loop through all the values, drawing a line segment at each point
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ySum += cur[1]
        # draw line from cur to (cur[0]-delX,cur[1]-delY)
        ax.plot([cur[0], cur[0]-delX], [cur[1], cur[1]-delY], c='b')
        cur = (cur[0] - delX, cur[1] - delY)
    # 随机猜测结果曲线
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print "the Area Under the Curve is: ", ySum * xStep


if __name__ == '__main__':
    # dataMat, classLabels = loadSimpData()
    # plotSimpData()

    # D = np.mat(np.ones((5, 1)) / 5) # 初始化均匀权值向量
    # bestStump, minError, bestClasEst = buildStump(dataMat, classLabels, D)
    # print(bestStump, minError, bestClasEst)

    # weakClassifiers, _ = adaBoostTrainDS(dataMat, classLabels, numIt=40)
    # print(weakClassifiers)
    # print(adaClassify([[5, 5], [0, 0]], weakClassifiers))

    testHorseData()