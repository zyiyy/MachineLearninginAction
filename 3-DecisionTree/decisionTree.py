# coding=utf-8


import math
from treePlotter import savePlot
import random


# 计算数据集的香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts:
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts.keys():
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * math.log(prob, 2)
    return shannonEnt

# 计算数据集的gini指数
def calcGini(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts:
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    gini = 1.0
    for key in labelCounts.keys():
        prob = float(labelCounts[key]) / numEntries
        gini -= prob ** 2
    return gini


def createDataSet():
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def splitDataSet(dataSet, axis, value):
    '''
    从数据集中取出某一列等于某个值的所有元素, 返回一个列表
    :param dataSet: 数据集
    :param axis: 列号
    :param value: 值
    :return: 满足条件的所有元素(去除了axis列)的列表
    '''
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureTosplit(dataSet, by='infoGain', max_features=None):
    '''
    基于信息增益的特征选择方式
    :param dataSet: 数据集
    :return: 最优特征
    '''
    features_index = []
    if by == 'random':
        if max_features == 'log2':
            features_log = int(math.log(len(dataSet[0]) - 1, 2))
            while len(features_index) <= features_log:
                index = random.randrange(len(dataSet[0]) - 1)
                if index not in features_index:
                    features_index.append(index)
        else:
            raise Exception('invalid max_features parameter.')
    else:
        numFeatures = len(dataSet[0]) - 1
        features_index = range(numFeatures)

    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    # bestFeature = - 100
    bestFeature = 0
    bestInfoGainRate = 0.0
    # bestFeatureRate = -100
    bestFeatureRate = 0

    baseGini = calcGini(dataSet)
    bestGiniGain = 0.0
    bestFeatureGini = 0

    for i in features_index:
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        featureEntropy = 0.0
        newGini = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
            featureEntropy -= prob * math.log(prob, 2)
            newGini += prob * calcGini(subDataSet)

        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i

        if featureEntropy == 0: # 在该列上只有一个取值
            # print(infoGain)
            # print(featList)
            # for j in range(len(dataSet)):
            #         print(dataSet[j])
            continue
        infoGainRate = infoGain / featureEntropy
        if infoGainRate > bestInfoGainRate:
            bestInfoGainRate = infoGainRate
            bestFeatureRate = i

        giniGain = baseGini - newGini
        if giniGain > bestGiniGain:
            bestGiniGain = giniGain
            bestFeatureGini = i

    if by == 'infoGain':
        # if bestInfoGain == 0:
        #     for j in range(len(dataSet)):
        #         print(dataSet[j])
        return bestFeature
    elif by == 'infoGainRate' or by == 'random':
        return bestFeatureRate
    elif by == 'gini':
        return bestFeatureGini
    else:
        raise Exception('invalid by parameter.')


def majorityCnt(classList):
    '''
    多数表决
    :param classList: 类别列表
    :return: 出现最多次数的类别
    '''
    classCount = {}
    for vote in classList:
        if vote not in classCount:
            classCount[vote] = 0
        else:
            classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=lambda item: item[1], reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels, max_depth=None, by='infoGain', max_features=None):
    '''
    创建决策树
    :param dataSet: 数据集
    :param labels: 特征(列)名
    :return: 决策树字典形式
    '''
    # 取数据集中所有样本的类别
    classList = [example[-1] for example in dataSet]
    if max_depth == 0:
        return majorityCnt(classList)
    if classList.count(classList[0]) == len(classList): # 所有类别全部相同则停止划分
        return classList[0]
    if len(dataSet[0]) == 1:    # 遍历完所有特征则返回出现次数最多的类别
        return majorityCnt(classList)
    # 选取最优的特征
    bestFeat = chooseBestFeatureTosplit(dataSet, by=by, max_features=max_features)
    # 最优特征名
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    myTree[bestFeatLabel]['majorClass'] = majorityCnt(classList)
    # 删除这个特征
    del(labels[bestFeat])
    feaValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(feaValues)
    # 循环最优特征的所有取值
    for value in uniqueVals:
        subLabels = labels[:]   # 复制一个切片, 浅拷贝, 里面不能有对象, 否则用深拷贝
        # print(splitDataSet(dataSet, bestFeat, value))
        if max_depth is not None:
            myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels, by=by, max_depth=max_depth - 1, max_features=max_features)
        else:
            myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels, by=by, max_features=max_features)
    return myTree


def classify(inputTree, featLables, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    # 找到特征列在原始数据集的位置
    featIndex = featLables.index(firstStr)
    classLabel = None
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            # print('ok' + testVec[featIndex])
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLables, testVec)
            else:
                classLabel = secondDict[key]
    if not classLabel:  # 没有目标分支, 提前终止
        # print('terminal' + testVec[featIndex])
        # print(firstStr)
        # print(inputTree)
        # classList = {}
        # searchTree(inputTree, classList)
        # classLabel = sorted(classList.items(), key=lambda item: item[1], reverse=True)[0][0]
        # print(classList)
        # print(classLabel)

        # print(inputTree)
        # print(inputTree.keys())
        # print(testVec[featIndex])
        # print(inputTree[firstStr]['majorClass'])
        classLabel = inputTree[firstStr]['majorClass']
    return classLabel


# 序列化操作, 以二进制保存决策树到文件
def storeTree(inputTree, filename):
    import pickle
    with open(filename, 'wb') as fw:
        pickle.dump(inputTree, fw)


# 读取决策树文件
def grabTree(filename):
    import pickle
    with open(filename, 'rb') as fr:
        return pickle.load(fr)


# 控制台打印整颗树
def printTree(inputTree, space=''):
    firstStr = inputTree.keys()[0]
    print(space + firstStr)
    secondDict = inputTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            print(space + '    ' + firstStr + ':' + str(key))
            printTree(secondDict[key], space + '    ')
        else:
            print(space + '    ' + firstStr + ':' + str(key))
            print(space + '    ' + secondDict[key])


# 输出树的类别列表
def searchTree(inputTree, classList):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            searchTree(secondDict[key], classList)
        else:
            classList.setdefault(secondDict[key], 0)
            classList[secondDict[key]] += 1


if __name__ == '__main__':
    dataSet, labels = createDataSet()

    # print(calcShannonEnt(dataSet))
    # dataSet[0][-1] = 'maybe'
    # print(calcShannonEnt(dataSet))

    # print(splitDataSet(dataSet, 0, 1))
    # print(chooseBestFeatureTosplit(dataSet))

    print(labels)
    myTree = grabTree('./storage/treeStorage.txt')
    savePlot(myTree, './storage/treeStorage.png')
    print(classify(myTree, labels, [0, 100]))
    print(classify(myTree, labels, [1, 0]))
    print(classify(myTree, labels, [1, 1]))
    classList = {}
    searchTree(myTree, classList)
    print(classList)

    with open('./data/lenses.txt', 'r') as fr:
        lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    # lensesTree = createTree(lenses, lensesLabels[:], by='infoGainRate')
    lensesTree = createTree(lenses, lensesLabels[:], by='random', max_features='log2')
    print(lensesLabels)
    print(lensesTree)
    storeTree(lensesTree, './storage/lensesTree.txt')
    savePlot(lensesTree, './storage/lensesTree.png')
    printTree(lensesTree, '')
    classList = {}
    searchTree(lensesTree, classList)
    print(classList)





