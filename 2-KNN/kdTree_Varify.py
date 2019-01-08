# coding=utf-8

from kdTree import kdTree
from kNN import file2matrix
import numpy as np
import operator

if __name__ == '__main__':
    datingDataMat, datingLabels = file2matrix('./data/datingTestSet.txt')
    inX = [40919, 8.326976, 0.953952]
    # 暴力计算欧式距离矩阵
    distanceMatrix = (np.tile(inX, (len(datingDataMat), 1)) - datingDataMat) ** 2
    distanceVector = np.sqrt(distanceMatrix.sum(axis=1))
    sorted_indices = distanceVector.argsort()
    kdRoot = kdTree(datingDataMat, datingLabels)
    classifierResult, nodeList = kdRoot.knn_algo(inX, k=200)
    count = 0
    for i in range(200):    # 与kdTree搜索到的前k个距离比较
        if operator.eq(list(nodeList[i][1]), list(datingDataMat[sorted_indices[i]])):
            count += 1
    print(count)