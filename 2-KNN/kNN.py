# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import os


# 简易数据集
def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def fourDotTest():
    dataSet, labels = createDataSet()
    # 新建图像
    plt.figure()
    # 散点图
    plt.scatter(dataSet[:, 0], dataSet[:, 1])
    # 设置图片, 坐标轴名
    plt.title('four dot')
    plt.xlabel('x1')
    plt.ylabel('x2')
    # 保存图片
    if not os.path.exists('four_dot.png'):
        plt.savefig('four_dot.png')
    plt.show()
    classify0([0.1, 0.2], dataSet, labels, 3)


# kNN分类器
# 输入 :
# inX     ： 待预测向量
# dataSet : 训练集
# labels  : 训练集标签
# k       : k个最近邻
# 输出 :
# 预测结果 : k个最近邻标签中出现次数最多的那个标签
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    # np.tile(X, (x,y)) : 重复X, 纵向重复x次, 横向重复y次; np.tile(X, x) : 重复X, 横向重复x次
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2  # 欧氏距离
    # sqDiffMat = np.abs(diffMat) # 曼哈顿距离
    # axis = 1 : 按行加; axis = 0 : 按列加
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    # 返回排序后的下标
    sortedDistIndicies = distances.argsort()
    classCount = dict()
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 字典按键 : c[0], 值 : c[1] 排序, 返回一个list
    sortedClassCount = sorted(classCount.items(), key=lambda c: c[1], reverse=True)
    # print(sortedClassCount)
    return sortedClassCount[0][0]


# 将文本记录转换为numpy
def file2matrix(filename):
    # 标签的字典
    love_dictionary = {'largeDoses': 3, 'smallDoses': 2, 'didntLike': 1}
    with open(filename, 'r') as f:
        arrayLines = f.readlines()
        numberOfLines = len(arrayLines)
        returnMat = np.zeros((numberOfLines, 3))
        classLabelVector = list()
        index = 0
        for line in arrayLines:
            # 去除行末的回车
            line = line.strip()
            # 按\t分隔line, 返回一个list
            listFromLine = line.split('\t')
            returnMat[index] = listFromLine[0:3]
            classLabelVector.append(int(love_dictionary[listFromLine[-1]]))
            index += 1
        return returnMat, classLabelVector


# 数值归一化, 消除不同特征具有不同量纲的问题, 全部缩放至(0, 1)
def autoNorm(dataSet):
    minVals = dataSet.min(axis=0)
    maxVals = dataSet.max(axis=0)
    ranges = maxVals - minVals
    dataSetSize = dataSet.shape[0]
    normDataSet = (dataSet - np.tile(minVals, (dataSetSize, 1))) / np.tile(ranges, (dataSetSize, 1))
    return normDataSet, ranges, minVals


# 标准化, 变成均值为0, 方差为1, 要原数据集的分布可近似为高斯分布效果较好, 应用在PCA降维
def standardization(dataSet):
    meanVals = dataSet.mean(axis=0)
    delta = np.sqrt(dataSet.var(axis=0))
    dataSetSize = dataSet.shape[0]
    stanDataSet = (dataSet - np.tile(meanVals, (dataSetSize, 1))) / np.tile(delta, (dataSetSize, 1))
    return stanDataSet, meanVals, delta


# 约会数据集kNN效果测评
def datingClassTest():
    datingDataMat, datingLabels = file2matrix('./data/datingTestSet.txt')
    hoRatio = 0.1
    # normDataMat = datingDataMat
    normDataMat, _, _ = autoNorm(datingDataMat)
    # normDataMat, _, _ = standardization(datingDataMat)
    m = normDataMat.shape[0]
    numTestVecs = int(hoRatio * m)
    errorCount = 0
    for i in range(numTestVecs):
        classifierResult = classify0(normDataMat[i], normDataMat[numTestVecs:], datingLabels[numTestVecs:], k=3)
        print('the classifier came back with : {}, the real answer is : {}'.format(classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print('ErrorCount is {}.'.format(errorCount))
    print('Accuracy is {}.'.format(1 - errorCount / float(numTestVecs)))


# 约会新样本kNN预测
def classifyPerson():
    resultDict = {3: 'largeDoses', 2: 'smallDoses', 1: 'didntLike'}
    ffMiles = float(raw_input('frequent flier miles earned per year?'))
    percentTats = float(raw_input('percentage of time spent playing video game?'))
    iceCream = float(raw_input('liters of ice cream consumed per year?'))
    datingDataMat, datingLabels = file2matrix('./data/datingTestSet.txt')
    normDataMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = (np.array([ffMiles, percentTats, iceCream]) - minVals) / ranges
    classfierResult = classify0(inArr, normDataMat, datingLabels, k=3)
    print('You will probably like this person : {}'.format(resultDict[classfierResult]))


# 将图像格式化处理为向量
def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    with open(filename, 'r') as f:
        for i in range(32):
            lineStr = f.readline()
            for j in range(32):
                returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


# 手写数字识别
def handwritingClassTest():
    hwLabels = list()
    trainingFileList = os.listdir('./data/digits/trainingDigits')
    m = len(trainingFileList)
    traningMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        hwLabels.append(int(fileNameStr.split('_')[0]))
        traningMat[i] = img2vector(os.path.join('./data/digits/trainingDigits', fileNameStr))
    print(traningMat.shape)

    testFileList = os.listdir('./data/digits/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        classNum = int(fileNameStr.split('_')[0])
        classifierResult = classify0(img2vector(os.path.join('./data/digits/testDigits', fileNameStr)),
                                    traningMat, hwLabels, k = 3)
        # print('the classifier came back with : {}, the real answer is : {}'.format(classifierResult, classNum))
        if classifierResult != classNum:
            print('the classifier came back with : {}, the real answer is : {}'.format(classifierResult, classNum))
            errorCount += 1.0
    print('ErrorCount is {}.'.format(errorCount))
    print('Accuracy is {}.'.format(1 - errorCount / float(mTest)))


if __name__ == '__main__':
    # fourDotTest()
    datingClassTest()
    # classifyPerson()
    # handwritingClassTest()


