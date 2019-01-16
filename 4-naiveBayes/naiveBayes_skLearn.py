# coding=utf-8


from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from naiveBayes import textParse, createVocabList, setOfwords2Vec, bagOfwords2VecMN
import random
import numpy as np


BNBModel = BernoulliNB()
'''
alpha: laplace平滑参数, 默认为1
binarize: 是否对特征进行二值化(0, 1), 默认为False, 即认为输入特征都是0或者1
fit_prior: 是否学习先验概率, 默认为True, 即各类的概率
class_prior: 设定各类的先验概率
'''
MModel = MultinomialNB()


def test(bag=False, prt=True):
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

    # 随机选择10个样本作为测试集, 剩下40个作为训练集
    trainingSet = range(50);
    testSet = []
    for i in range(10):
        randIndex = random.randrange(len(trainingSet))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])

    if not bag:
        trainMat = []; trainClasses = []
        for docIndex in trainingSet:
            trainMat.append(setOfwords2Vec(vocabList, docList[docIndex]))
            trainClasses.append(classList[docIndex])
        BNBModel.fit(trainMat, trainClasses)
        errorCount = 0
        errorClassList = []
        for docIndex in testSet:
            wordVector = np.array(setOfwords2Vec(vocabList, docList[docIndex])).reshape(1, -1)
            predict = BNBModel.predict(wordVector)
            if predict != classList[docIndex]:
                errorClassList.append(classList[docIndex])
                errorCount += 1
        errorRate = float(errorCount) / len(testSet)
        if prt:
            print('the error rate is {}.'.format(errorRate))
        return errorRate, errorClassList
    else:
        trainMat = []; trainClasses = []
        for docIndex in trainingSet:
            trainMat.append(bagOfwords2VecMN(vocabList, docList[docIndex]))
            trainClasses.append(classList[docIndex])
        MModel.fit(trainMat, trainClasses)
        errorCount = 0
        errorClassList = []
        for docIndex in testSet:
            wordVector = np.array(bagOfwords2VecMN(vocabList, docList[docIndex])).reshape(1, -1)
            predict = MModel.predict(wordVector)
            if predict != classList[docIndex]:
                errorClassList.append(classList[docIndex])
                errorCount += 1
        errorRate = float(errorCount) / len(testSet)
        if prt:
            print('the error rate is {}.'.format(errorRate))
        return errorRate, errorClassList


def testRepeat(n, bag=False, prt=True):
    errorRateR = 0.0
    errorClassListR = []
    for i in range(n):
        errorRate, errorClassList = test(bag, prt)
        errorRateR += errorRate
        errorClassListR.extend(errorClassList)
    errorRateR /= n
    return errorRateR, errorClassListR


if __name__ == '__main__':
    # 使用词袋模型
    errorRateR, errorClassListR = testRepeat(1000, bag=True, prt=False)
    print(errorRateR, np.array(errorClassListR).sum(), len(errorClassListR) - np.array(errorClassListR).sum())
    # 不使用词袋模型
    errorRateR, errorClassListR = testRepeat(1000, bag=False, prt=False)
    print(errorRateR, np.array(errorClassListR).sum(), len(errorClassListR) - np.array(errorClassListR).sum())


