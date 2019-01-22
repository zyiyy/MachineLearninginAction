# coding=utf-8


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from logisticRegression import sigmoid


def printLoss():
    '''
    Logistic回归参数优化是凸优化, 可视化两个样例的损失函数
    :return:
    '''
    x = [[1, 2], [1, 3]]
    y = [1, 0]
    fig = plt.figure()
    ax = Axes3D(fig)
    W1 = np.arange(-5, 5, 0.01)
    W2 = np.arange(-5, 5, 0.01)
    W1, W2 = np.meshgrid(W1, W2)
    Z = 1 - sigmoid(W1 * x[0][0] + W2 * x[0][1]) + sigmoid(W1 * x[1][0] + W2 * x[1][1])
    Z = - Z
    ax.plot_surface(W1, W2, Z)
    plt.show()


if __name__ == '__main__':
    printLoss()