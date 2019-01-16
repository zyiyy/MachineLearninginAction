# coding=utf-8


import numpy as np
import random
import feedparser


# 加载简易数据集
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    # 辱骂文字, 1 : 是 0 ：不是
    return postingList,classVec


# 产生词汇表
def createVocabList(dataSet):
    vocabSet = set()
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


# 将文档转化成词集模型, 在词集模型中, 每个词只能出现一次
def setOfwords2Vec(vocabList, inputSet):
    '''
    基于词汇表产生词向量, 词向量与词汇表同长度, 若词汇表中第i个词出现, 词向量相应位置置1, 否则为0
    :param vocabList: 词汇表
    :param inputSet: 原文档
    :return: 词向量
    '''
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
    return returnVec


# 将文档转化成词袋模型, 在词袋模型中, 每个词可以出现多次
def bagOfwords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


# 基于词集模型的训练函数
def trainNB0(trainMatrix, trainCategory):
    # 计算训练集大小
    numTrainDocs = len(trainMatrix)
    # 计算词向量长度
    numWords = len(trainMatrix[0])
    # 计算所有文档中各类文档(0, 1)所占的比例
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # 拉普拉斯平滑
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    # 机器学习实战中此处分母计算有误, 标签为1情况下'stupid'出现的条件概率正确答案是1
    p1Denom = sum(trainCategory)
    p0Denom = len(trainCategory) - p1Denom
    p0Denom += 2
    p1Denom += 2
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
        else:
            p0Num += trainMatrix[i]
    p1Vect = np.log2(p1Num / float(p1Denom))
    p0Vect = np.log2(p0Num / float(p0Denom))
    return p0Vect, p1Vect, pAbusive


# 基于词袋模型的训练函数, 与词袋模型配套使用
def trainNB0Bag(trainMatrix, trainCategory):
    # 计算训练集大小
    numTrainDocs = len(trainMatrix)
    # 计算词向量长度
    numWords = len(trainMatrix[0])
    # 计算所有文档中各类文档(0, 1)所占的比例
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # 拉普拉斯平滑
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2
    p1Denom = 2
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += np.array(trainMatrix[i]).sum()
        else:
            p0Num += trainMatrix[i]
            p0Denom += np.array(trainMatrix[i]).sum()
    p1Vect = np.log2(p1Num / float(p1Denom))
    p0Vect = np.log2(p0Num / float(p0Denom))
    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    '''
    使用NaiveBayes进行分类, 这里的概率不是计算真正的概率, 在朴素贝叶斯中为了简化计算只比较了不同类别的对数概率大小
    详细说明：由于不同类别概率计算公式的分母相同, 只需要比较分子大小, 又因为log是单调函数, 只需要比较log(p0)与log(p1)的相对大小,
    因为分子是连乘, 这样可以简化计算
    :param vec2Classify:
    :param p0Vec: 0类词概率向量
    :param p1Vec: 1类词概率向量
    :param pClass1: 1类概率
    :return:
    '''
    p1 = sum(vec2Classify * p1Vec) + np.log2(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log2(1 - pClass1)
    predict = 1 if p1 > p0 else 0
    return predict


def testingNB():
    # 获取训练集, 类别
    listOPosts, listClasses = loadDataSet()
    # 产生词汇表
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    # 将训练集转化成词集模型: 只关心出现与否, 不关心出现次数
    for postinDoc in listOPosts:
        trainMat.append(setOfwords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(trainMat, listClasses)
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfwords2Vec(myVocabList, testEntry))
    print('{} classified as: {}.'.format(testEntry, classifyNB(thisDoc, p0V, p1V, pAb)))
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfwords2Vec(myVocabList, testEntry))
    print('{} classified as: {}.'.format(testEntry, classifyNB(thisDoc, p0V, p1V, pAb)))


def textParse(bigString):
    '''
    # 使用正则表达式来切分句子, 去掉长度小于2的词并全部转换为小写
    :param bigString: 输入长字符串
    :return: 切分后的list
    '''
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spamTest(bag=False, prt=True):
    # docList : 切分后的数据集 classList : 类别
    docList = []; classList = []; fullText = []
    # 导入文件夹spam和ham下的文本文件并将它们解析为词列表
    for i in range(1, 26):
        wordList = textParse(open('./data/email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('./data/email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    # 产生词汇表
    vocabList = createVocabList(docList)

    # top5Words = clacMostFreq(vocabList, fullText, n=5)
    # # 去掉出现前5高频词
    # for pairW in top5Words:
    #     if pairW[0] in vocabList: vocabList.remove(pairW[0])

    # 随机选择10个样本作为测试集, 剩下40个作为训练集
    trainingSet = range(50); testSet = []
    for i in range(10):
        randIndex = random.randrange(len(trainingSet))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    # print(trainingSet)
    # print(testSet)
    if not bag:
        trainMat = []; trainClasses = []
        for docIndex in trainingSet:
            trainMat.append(setOfwords2Vec(vocabList, docList[docIndex]))
            trainClasses.append(classList[docIndex])
        p0V, p1V, pSpam = trainNB0(trainMat, trainClasses)
        errorCount = 0
        errorClassList = []
        for docIndex in testSet:
            wordVector = setOfwords2Vec(vocabList, docList[docIndex])
            predict = classifyNB(wordVector, p0V, p1V, pSpam)
            if predict != classList[docIndex]:
                # print(classList[docIndex], docList[docIndex])
                errorClassList.append(classList[docIndex])
                errorCount += 1
            # print(predict, classList[docIndex])
        errorRate = float(errorCount) / len(testSet)
        if prt:
            print('the error rate is {}.'.format(errorRate))
        return errorRate, errorClassList
    else:
        trainMat = []; trainClasses = []
        for docIndex in trainingSet:
            trainMat.append(bagOfwords2VecMN(vocabList, docList[docIndex]))
            trainClasses.append(classList[docIndex])
        p0V, p1V, pSpam = trainNB0Bag(trainMat, trainClasses)
        errorCount = 0
        errorClassList = [] # 错误样本类别列表
        for docIndex in testSet:
            wordVector = bagOfwords2VecMN(vocabList, docList[docIndex])
            predict = classifyNB(wordVector, p0V, p1V, pSpam)
            if predict != classList[docIndex]:
                # print(classList[docIndex], docList[docIndex])
                errorClassList.append(classList[docIndex])
                errorCount += 1
            # print(predict, classList[docIndex])
        errorRate = float(errorCount) / len(testSet)
        if prt:
            print('the error rate is {}.'.format(errorRate))
        return errorRate, errorClassList


def spamTestRepeat(n, bag=False, prt=True):
    errorRateR = 0.0
    errorClassListR = []
    for i in range(n):
        errorRate, errorClassList = spamTest(bag, prt)
        errorRateR += errorRate
        errorClassListR.extend(errorClassList)
    errorRateR /= n
    return errorRateR, errorClassListR


# 计算前n个高频词
def clacMostFreq(vocabList, fullText, n=10):
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=lambda item: item[1], reverse=True)
    return sortedFreq[:n]


if __name__ == '__main__':
    # testingNB()
    # spamTest()
    # spamTest(True)

    # 使用词袋模型
    errorRateR, errorClassListR = spamTestRepeat(1000, bag=True, prt=False)
    print(errorRateR, np.array(errorClassListR).sum(), len(errorClassListR) - np.array(errorClassListR).sum())
    # 不使用词袋模型
    errorRateR, errorClassListR = spamTestRepeat(1000, bag=False, prt=False)
    print(errorRateR, np.array(errorClassListR).sum(), len(errorClassListR) - np.array(errorClassListR).sum())








