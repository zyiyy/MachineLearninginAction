# coding=utf-8
import numpy as np
import kNN
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # 使用样本的第二列和第三列展示数据
    fig = plt.figure(1)
    # add_subplot(m, n, i)把m * n张图片画在一个平面上, 绘制第i张图片
    ax = fig.add_subplot(1, 1, 1)
    datingDataSet, datingLabels = kNN.file2matrix('../data/datingTestSet.txt')
    # x, y, s, c : x, y, size, color
    ax.scatter(datingDataSet[:, 1], datingDataSet[:, 2], s=30.0 * np.array(datingLabels), c=15.0 * np.array(datingLabels))
    # 控制横纵坐标[xmin, xmax, ymin, ymax]
    ax.axis([-2, 25, -0.2, 2.0])
    plt.xlabel('Percentage of Time Spent Playing Video Games')
    plt.ylabel('Liters of Ice Cream Consumed Per Week')
    plt.show()
    plt.close(1)

    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.scatter(datingDataSet[:, 0], datingDataSet[:, 1], s=30.0 * np.array(datingLabels), c=15.0 * np.array(datingLabels))
    plt.xlabel('Frequent Flyier Miles Earned Per Year')
    plt.ylabel('Percentage of Time Spent Playing Video Games')
    plt.show()
    plt.close(2)
