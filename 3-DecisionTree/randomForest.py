# coding=utf-8

from sklearn.ensemble import RandomForestClassifier
import random
import pandas as pd
import numpy as np
from decisionTree import createTree, printTree, classify, searchTree


def subSample(dataSet):
    '''
    有放回随机采样, 每次采与原数据集相同大小样本, 约1/3样本不被取到
    :param dataSet: 输入大小为m的数据集
    :return: 大小为m的样本
    '''
    sample = []
    while len(sample) < len(dataSet):
        sample.append(dataSet[random.randrange(len(dataSet))])
    return sample


def randomForest(dataSet, labels, num_trees=2, max_depth=None, by='random', max_features='log2'):
    decisionTrees = []
    for i in range(num_trees):
        subDataSet = subSample(dataSet)
        tree = createTree(subDataSet, labels[:], max_depth=max_depth, by=by, max_features=max_features)
        decisionTrees.append(tree)
    return decisionTrees


def randomForestClassify(myForest, featLables, testVec):
    classCount = {}
    for tree in myForest:
        firstStr = tree.keys()[0]
        secondDict = tree[firstStr]
        # 找到特征列在原始数据集的位置
        featIndex = featLables.index(firstStr)
        classLabel = None
        for key in secondDict.keys():
            if testVec[featIndex] == key:
                if type(secondDict[key]).__name__ == 'dict':
                    classLabel = classify(secondDict[key], featLables, testVec)
                else:
                    classLabel = secondDict[key]
        if not classLabel:  # 没有目标分支, 提前终止
            classList = {}
            searchTree(tree, classList)
            classLabel = sorted(classList.items(), key=lambda item: item[1], reverse=True)[0][0]
        classCount.setdefault(classLabel, 0)
        classCount[classLabel] += 1
    return sorted(classCount.items(), key=lambda item: item[1], reverse=True)[0][0]


# 测试结果, 准确率提升了约百分之一
def test(myForest, labels, data, prt=True):
    '''
    测试分类器效果
    :param myForest: 分类器
    :param labels: 特征列
    :param data: 二维list
    :return: 准确率, 混淆矩阵
    '''
    count = 0; escape = 0
    ac = 0; acpc = 0; au = 0; aupu = 0
    for i in range(len(data)):
        dataVec = list(data[i])
        predict = randomForestClassify(myForest, labels, dataVec)
        if predict == None:
            escape += 1
            continue
        if dataVec[-1] == ' >50K':
            au += 1
            if predict == ' >50K':
                aupu += 1
        else:
            ac += 1
            if predict == ' <=50K':
                acpc += 1
        if predict == dataVec[-1]:
            count += 1
    accuracy = (count + escape) / float(len(data))
    print('escape: {}'.format(escape))
    confuse_matrix = np.array([[acpc, ac-acpc], [au - aupu, aupu]])
    if prt:
        print('accuracy: {}'.format(accuracy))
        print(confuse_matrix)
    return accuracy, confuse_matrix


if __name__ == '__main__':
    # with open('./data/lenses.txt', 'r') as fr:
    #     lensesData = [inst.strip().split('\t') for inst in fr.readlines()]
    # lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    #
    # rForest = randomForest(lensesData, lensesLabels)
    # for tree in rForest:
    #     printTree(tree)

    data = pd.read_csv('./data/adults/adult.data', header=None,
                       names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                              'occupation',
                              'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                              'native-country', 'salary'])
    data_test = pd.read_csv('./data/adults/adult.test', header=None,
                            names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                                   'occupation',
                                   'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                                   'native-country', 'salary'])
    # 第一行没有数据, 删去第一行
    data_test.dropna(inplace=True)
    # 删除salary列最后的'.', 与data保持一致
    data_test['salary'] = data_test['salary'].map(lambda s: s[:-1])
    # 把data_test中的age列转成float型, 与data保持一致
    data_test['age'] = data_test['age'].map(lambda x: float(x))
    labels = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
              'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
    # 拼接训练集和测试集
    data = data.append(data_test)

    # 把data中的连续值转化为离散值
    labels_mean_dict = {'age': 37, 'fnlwgt': 1.781445e+05, 'education-num': 10, 'capital-gain': 0, 'capital-loss': 0,
                        'hours-per-week': 40}
    for label in labels:
        if type(data[label][0]).__name__ == 'float64':
            data[label] = data[label].map(lambda x: str(x > labels_mean_dict[label]))

    # 划分训练集, 测试集
    data_train = data.head(len(data) - len(data_test)).values.tolist()
    data_test = data.tail(len(data_test)).values.tolist()

    rForest = randomForest(data_train, labels, num_trees=100, max_depth=8)
    # for tree in rForest:
    #     printTree(tree)
    # 分别在训练集,测试集上测试模型的效果
    test(rForest, labels, np.array(data_test[23]).reshape(1, -1).tolist(), prt=True)
    accuracy_train, confuse_matrix_train = test(rForest, labels, data_train, prt=True)
    accuracy_test, confuse_matrix_test = test(rForest, labels, data_test, prt=True)

