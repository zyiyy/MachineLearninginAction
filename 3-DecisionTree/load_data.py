# coding=utf-8


import pandas as pd
import numpy as np
from decisionTree import createTree, storeTree, grabTree, classify, printTree, searchTree
from treePlotter import savePlot, getNumLeafs
# pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth',100)


# 返回特征列的所有取值
def labels_dict(data, labels):
    label_dict = {}
    for i in range(data.shape[1] - 1):
        label_dict.setdefault(labels[i], set())
        for j in range(data.shape[0]):
            label_dict[labels[i]].add(data[labels[i]].iloc[j])
    return label_dict


# 打印DataFrame的统计结果
def describe(data, prt=True):
    '''
    DataFrame.describe()

    数值型
    count : 计数
    mean  : 平均值
    std   : 标准差
    min   : 最小值
    25%   : 百分位数Q1
    50%   : 百分位数Q2, 中位数
    75%   : 百分位数Q3
    max   : 最大值

    非数值型
    count  : 计数
    unique : 不同取值个数
    top    : 出现最高频率的值
    freq   : 最高频次

    参数
    include : 列表, 包含的列
    exclude : 列表, 不包含的列

    '''
    num_describe = data.describe(include=[np.number])
    obj_describe = data.describe(include=[np.object])
    if prt:
        print(num_describe)
        print(obj_describe)
    return num_describe, obj_describe

# 测试结果
def test(inputTree, labels, data, prt=True):
    '''
    测试分类器效果
    :param inputTree: 分类器
    :param labels: 特征列
    :param data: 二维list
    :return: 准确率, 混淆矩阵
    '''
    count = 0; escape = 0
    ac = 0; acpc = 0; au = 0; aupu = 0
    for i in range(len(data)):
        dataVec = list(data[i])
        predict = classify(inputTree, labels, dataVec)
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


def load_tree(data, labels, max_depth=None, by='infoGain', max_features=None, grab=False, store=False, prt=False):
    if grab:
        myTree = grabTree('adults.txt')
    else:
        myTree = createTree(data, labels[:], max_depth=max_depth, by=by, max_features=max_features) # 复制一个切片, 防止labels被修改
    if store:
        storeTree(myTree, 'adults.txt')
        savePlot(myTree, 'adults.png')
    if prt:
        printTree(myTree, '')
    return myTree


if __name__ == '__main__':
    data = pd.read_csv('./data/adults/adult.data', header=None, names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                              'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'salary'])
    data_test = pd.read_csv('./data/adults/adult.test', header=None, names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                              'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'salary'])
    # 第一行没有数据, 删去第一行
    data_test.dropna(inplace=True)
    # 删除salary列最后的'.', 与data保持一致
    data_test['salary'] = data_test['salary'].map(lambda s: s[:-1])
    # 把data_test中的age列转成float型, 与data保持一致
    data_test['age'] = data_test['age'].map(lambda x: float(x))
    labels = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                              'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
    # 拼接训练集和测试集
    data = data.append(data_test, ignore_index=True)

    # 把data中的连续值转化为离散值
    # num_describe, _ = describe(data, prt=True)
    # label_dict = labels_dict(data, labels)
    # for key, value in label_dict.items():
    #     print(key, value)
    labels_mean_dict = {'age': 37, 'fnlwgt': 1.781445e+05, 'education-num': 10, 'capital-gain': 0, 'capital-loss': 0, 'hours-per-week': 40}
    for label in labels:
        if type(data[label][0]).__name__ == 'float64':
            # print(label)
            data[label] = data[label].map(lambda x: str(x > labels_mean_dict[label]))
    data.to_csv('./data/adults/processed_adults.csv')

    # print(data[0:10])

    # # 仅看非数值列
    # data = data[['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'salary']]
    # labels = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'salary']

    # 划分训练集, 测试集
    data_train = data.head(len(data) - len(data_test)).values.tolist()
    data_test = data.tail(len(data_test)).values.tolist()

    # 加载模型
    myTree = load_tree(data_train, labels, max_depth=5, by= 'infoGainRate', grab=False, store=True, prt=False)
    # myTree = load_tree(data_train, labels, max_depth=5, by='random', max_features='log2', grab=False, store=True, prt=False)

    # 分别在训练集,测试集上测试模型的效果
    test(myTree, labels, np.array(data_test[23]).reshape(1, -1).tolist(), prt=True)
    accuracy_train, confuse_matrix_train = test(myTree, labels, data_train, prt=True)
    accuracy_test, confuse_matrix_test = test(myTree, labels, data_test, prt=True)













