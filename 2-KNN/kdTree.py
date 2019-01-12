# coding=utf-8

import numpy as np
import time
import matplotlib.pyplot as plt
from kNN import file2matrix, classify0


class Node(object):
    def __init__(self, item=None, label=None, dim=None, parent=None, left_child=None, right_child=None):
        '''
        :param item: 特征向量 X
        :param label: 标签 y
        :param dim: 切分维度
        :param parent: 父结点
        :param left_child: 左孩子
        :param right_child: 右孩子
        '''
        self.item = item
        self.label = label
        self.dim = dim
        self.parent = parent
        self.left_child = left_child
        self.right_child = right_child


class kdTree(object):
    def __init__(self, aList, labelList):
        self.__length = 0   # 树的结点总数, 只读属性
        self.__root = self.__create(aList, labelList)   # 树的根结点, 只读属性
        self.index = 0

    def __create(self, dataList, labelList, parentNode=None):
        '''
        构造kd树
        :param dataList: 数据集, m行表示样本数量, n代表特征维度
        :param labelList: 标签集, 大小为m
        :param parentNode: 父亲结点
        :return: kdTree根结点
        '''
        dataArray = np.array(dataList)
        m, n = dataArray.shape
        if m == 0:
            return None
        labelArray = np.array(labelList).reshape(m, 1)

        # 选择方差最大的特征作为切分超平面
        varValue = [np.var(dataArray[:, col]) for col in range(n)]
        maxVarIndex = np.array(varValue).argsort()[-1]  # 保存方差最大的特征列号

        # 按最大方差特征列排序, 取中位数
        sorted_indices = dataArray[:, maxVarIndex].argsort()
        mid_item_index = sorted_indices[m // 2]

        if m == 1:  # 该结点是叶子结点
            self.__length += 1
            return Node(item=dataArray[mid_item_index], label=labelArray[mid_item_index], parent=parentNode, left_child=None, right_child=None)

        # 该结点是中间结点
        node = Node(item=dataArray[mid_item_index], label=labelArray[mid_item_index], dim=maxVarIndex, parent=parentNode)

        # 递归构造左右子树
        left_data = dataArray[sorted_indices[:m // 2]]
        left_label = labelArray[sorted_indices[:m // 2]]
        left_child = self.__create(left_data, left_label, parentNode=node)
        if m == 2:  # 只有左子树
            right_child = None
        else:
            right_data = dataArray[sorted_indices[m // 2 + 1:]]
            right_label = labelArray[sorted_indices[m // 2 + 1:]]
            right_child = self.__create(right_data, right_label, node)
        node.left_child = left_child
        node.right_child = right_child
        self.__length += 1
        return node

    @property
    def length(self):
        return self.__length

    @property
    def root(self):
        return self.__root

    def transfer_dict(self, node):
        '''
        查看kd树结构
        :param node: 根结点
        :return: 字典嵌套格式的kd树
        实例:

        input:
        dataList = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
        labelList = np.array([0, 0, 0, 0, 1, 1])
        kdRoot = kdTree(dataList, labelList)
        kdRoot.transfer_dict(kdRoot.root)

        output:
        {
        (7, 2):
        {
        'dim': 0,
        'left_child':
            {
            (5, 4):
            {
            'dim': 1,
            'left_child':
                {
                (2, 3):
                {
                'dim': None,
                'left_child': None,
                'right_child': None,
                'parent': array([5, 4]),
                'label': array([0])
                }
                },
            'right_child':
                {
                (4, 7):
                {
                'dim': None,
                'left_child': None,
                'right_child': None,
                'parent': array([5, 4]),
                'label': array([0])
                }
                },
            'parent': array([7, 2]),
            'label': array([0])
            }
            },
        'right_child':
            {
            (9, 6):
            {
            'dim': 1,
            'left_child':
                {
                (8, 1):
                {
                'dim': None,
                'left_child': None,
                'right_child': None,
                'parent': array([9, 6]),
                'label': array([1])
                }
                },
            'right_child': None,
            'parent': array([7, 2]),
            'label': array([0])
            }
            },
        'parent': None,
        'label': array([1])
        }
        }
        '''
        if node == None:
            return None
        kd_dict = {}
        kd_dict[tuple(node.item)] = {}
        kd_dict[tuple(node.item)]['label'] = node.label[0]
        kd_dict[tuple(node.item)]['dim'] = node.dim
        kd_dict[tuple(node.item)]['parent'] = (node.parent.item) if node.parent else None
        kd_dict[tuple(node.item)]['left_child'] = self.transfer_dict(node.left_child)
        kd_dict[tuple(node.item)]['right_child'] = self.transfer_dict(node.right_child)
        return kd_dict

    def find_cur_nearest_neighbor(self, item):
        '''
        寻找当前最近的叶子结点, 不一定是真正的最近
        :param item: 输入样本
        :return: 样本的当前最近叶子结点
        '''
        if self.length == 0:
            return None
        node = self.root
        if self.length == 1:    # 树只有一个结点
            return node

        while True:
            if node.left_child == None:
                return node
            cur_dim = node.dim
            if item[cur_dim] <= node.item[cur_dim]:
                node = node.left_child
            else:
                if node.right_child == None:
                    node = node.left_child
                else:
                    node = node.right_child

    def knn_algo(self, item, k=1):
        item = np.array(item)
        node = self.find_cur_nearest_neighbor(item)
        if node == None:
            return None
        node_list = []
        cur_distance = np.sqrt(sum((item - node.item) ** 2))
        node_list.append([cur_distance, tuple(node.item), node.label[0]])

        # 递归向上回退
        while True:
            if node == self.root:   # 回到了kd树根结点
                break
            parent = node.parent
            par_distance = np.sqrt(sum((item - parent.item) ** 2))
            node_list.sort()
            cur_distance = node_list[-1][0] if k >= len(node_list) else node_list[k - 1][0]
            # 判断父结点到测试点的距离, 若小于当前最短距离, 则更新当前最短距; 若node_list中结点数不足k, 也把该点信息加入
            if k > len(node_list) or par_distance < cur_distance:
                node_list.append([par_distance, tuple(parent.item), parent.label[0]])
                node_list.sort()
                cur_distance = node_list[-1][0] if k >= len(node_list) else node_list[k - 1][0]

            # 判断父结点的另一个子树与以测试点为圆心、当前k个最短距离中最大距离为半径的圆是否有交集
            if k > len(node_list) or abs(item[parent.dim] - parent.item[parent.dim]) < cur_distance:
                # node是parent的左子树, 则搜索域是parent的右子树
                other_child = parent.left_child if parent.left_child != node else parent.right_child

                if other_child != None:
                    self.search(item, other_child, node_list, k)

            node = parent

        # 取前k个邻居中出现最多的分类标签作为预测结果
        label_dict = {}
        node_list = node_list[:k]
        for element in node_list:
            if element[2] in label_dict:
                label_dict[element[2]] += 1
            else:
                label_dict[element[2]] = 1
        sorted_label = sorted(label_dict.items(), key=lambda item:item[1], reverse=True)
        return sorted_label[0][0], node_list

    def search(self, item, node, nodeList, k):
        nodeList.sort()  # 对结点列表按距离排序
        least_dis = nodeList[-1][0] if k >= len(nodeList) else nodeList[k - 1][0]

        if node.left_child == None and node.right_child == None:  # 叶结点
            dis = np.sqrt(sum((item - node.item) ** 2))
            if k > len(nodeList) or dis < least_dis:
                nodeList.append([dis, tuple(node.item), node.label[0]])
            return

        self.search(item, node.left_child, nodeList, k)

        # 每次进行比较前都更新nodelist数据
        nodeList.sort()  # 对结点列表按距离排序
        least_dis = nodeList[-1][0] if k >= len(nodeList) else nodeList[k - 1][0]

        # 比较根结点
        dis = np.sqrt(sum((item-node.item)**2))
        if k > len(nodeList) or dis < least_dis:
            nodeList.append([dis, tuple(node.item), node.label[0]])

        if node.right_child != None:
            self.search(item, node.right_child, nodeList, k)

        nodeList.sort()
        return


if __name__ == '__main__':
    dataList = np.array([[2, 4], [5, 1], [3, 6], [7, 3], [6, 4.3], [2, 1], [1, 7]])
    labelList = np.array([0, 0, 0, 0, 1, 1, 1])

    kdRoot = kdTree(dataList, labelList)
    print(kdRoot.transfer_dict(kdRoot.root))
    print(kdRoot.length)
    label, nodeList = kdRoot.knn_algo([6, 3.8], k=1)
    print(nodeList)

    datingDataMat, datingLabels = file2matrix('./data/datingTestSet.txt')
    from sklearn import preprocessing
    datingDataMat = preprocessing.MinMaxScaler().fit_transform(datingDataMat)
    numTest = int(0.1 * datingDataMat.shape[0])
    kdRoot = kdTree(datingDataMat[numTest:], datingLabels[numTest:])
    errorCount = 0
    for i in range(numTest):
        classifierResult, nodeList = kdRoot.knn_algo(datingDataMat[i], k=3)
        print('the classifier came back with : {}, the real answer is : {}'.format(classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print(errorCount)





