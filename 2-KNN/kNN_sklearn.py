# coding=utf-8


from sklearn import neighbors
from sklearn import preprocessing
from kNN import file2matrix, img2vector
import os
import numpy as np

if __name__ == '__main__':
    model = neighbors.KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='kd_tree', n_jobs=1)
    '''
    ---
    参数：
    n_neighbors : 邻居数目, int型常量
    weights : 最近邻权重计算方式, uniform各个邻居的权重相同
    algorithm : 搜索算法, kdTree, ballTree
    metric : 距离度量, 默认欧式距离(p=2)
    n_jobs : 并行任务数
    '''
    # datingDataMat, datingLabels = file2matrix('./data/datingTestSet.txt')
    # # 归一化
    # datingDataMat = preprocessing.MinMaxScaler().fit_transform(datingDataMat)
    # numTest = int(0.1 * datingDataMat.shape[0])
    # model.fit(datingDataMat[numTest:], datingLabels[numTest:])
    # errorCount = 0
    # classifierResult = model.predict(datingDataMat[:numTest])
    # for i in range(numTest):
    #     print('the classifier came back with : {}, the real answer is : {}'.format(classifierResult[i], datingLabels[i]))
    #     if classifierResult[i] != datingLabels[i]:
    #         errorCount += 1.0
    # print(errorCount)

    hwLabels = list()
    trainingFileList = os.listdir('./data/digits/trainingDigits')
    m = len(trainingFileList)
    traningMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        hwLabels.append(int(fileNameStr.split('_')[0]))
        traningMat[i] = img2vector(os.path.join('./data/digits/trainingDigits', fileNameStr))
    model.fit(traningMat, hwLabels)
    print(traningMat.shape)

    testFileList = os.listdir('./data/digits/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        classNum = int(fileNameStr.split('_')[0])
        classifierResult = model.predict(img2vector(os.path.join('./data/digits/testDigits', fileNameStr)))
        # print('the classifier came back with : {}, the real answer is : {}'.format(classifierResult, classNum))
        if classifierResult != classNum:
            print('the classifier came back with : {}, the real answer is : {}'.format(classifierResult, classNum))
            errorCount += 1.0
    print('ErrorCount is {}.'.format(errorCount))
    print('Accuracy is {}.'.format(1 - errorCount / float(mTest)))