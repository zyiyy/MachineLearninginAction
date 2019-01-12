#coding=utf-8


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_classification
import time

# X为样本特征, y为样本类别输出, 共1000个样本, 每个样本2个特征, 没有冗余特征, 每个类别一个簇, 输出3个类别
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_clusters_per_class=1, n_classes=3)

plt.scatter(X[:, 0], X[:, 1], marker='*', c=y)
plt.show()

print('skLearn start:' + time.ctime())
from sklearn import neighbors
clf = neighbors.KNeighborsClassifier(n_neighbors=15, weights='uniform')
clf.fit(X, y)

from matplotlib.colors import ListedColormap
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# 训练集边界
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

# 生成随机数据作为测试集, 作用类似二重循环
x1, x2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02), np.arange(x2_min, x2_max, 0.02))

predict = clf.predict(np.c_[x1.ravel(), x2.ravel()])

predict = predict.reshape(x1.shape)
# 画出测试集数据
plt.figure()
plt.pcolormesh(x1, x2, predict, cmap=cmap_light)
# 画出训练集数据
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
plt.title("3-Class classification (k = 15, weights = 'uniform')" )
plt.show()
print('skLearn end:' + time.ctime())


from kdTree import kdTree
kdRoot = kdTree(X, y)
# print(kdRoot.length)
print('create tree done:' + time.ctime())
predict = []
for x1_test, x2_test in np.c_[x1.ravel(), x2.ravel()]:
    classifierResult, _ = kdRoot.knn_algo([x1_test, x2_test], k=15)
    predict.append(classifierResult)
predict = np.array(predict).reshape(x1.shape)
# 画出测试集数据
plt.figure()
plt.pcolormesh(x1, x2, predict, cmap=cmap_light)
# 画出训练集数据
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
plt.title("3-Class classification (k = 15, weights = 'uniform')" )
plt.show()
print('kd tree done:' + time.ctime())

from kNN import classify0
predict = []
for x1_test, x2_test in np.c_[x1.ravel(), x2.ravel()]:
    classifierResult = classify0([x1_test, x2_test], X, y, k=15)
    predict.append(classifierResult)
predict = np.array(predict).reshape(x1.shape)
# 画出测试集数据
plt.figure()
plt.pcolormesh(x1, x2, predict, cmap=cmap_light)
# 画出训练集数据
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
plt.title("3-Class classification (k = 15, weights = 'uniform')" )
plt.show()
print(time.ctime())
