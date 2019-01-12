# coding=utf-8


import matplotlib.pyplot as plt


# 定义文本框和箭头格式
decisionNode = dict(boxstyle='sawtooth', fc='0.8')
leafNode = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle='<-')


# 绘制带箭头的注解
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va='center', ha='center', bbox=nodeType, arrowprops=arrow_args)


# def createPlot():
#     # 创建一个新图形
#     fig = plt.figure(1, facecolor='white')
#     # 清空绘图区
#     fig.clf()
#     # 声明绘图区
#     createPlot.ax1 = plt.subplot(1, 1, 1, frameon=False)
#     plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
#     plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
#     plt.show()


# 获取叶节点的数目
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        # 若子节点仍然是字典
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


# 获取树的层数
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


# 预先存储的树的信息
def retrieveTree(i):
    listOfTrees = [
        {'no surfacing':
             {0: 'no', 1: {'flippers':
                               {0: 'no', 1: 'yes'}
                          }
             }
        },
        {'no surfacing':
             {0: 'no', 1: {'flippers':
                               {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}
                          }
             }

        }
    ]
    return listOfTrees[i]


# 在父子结点间填充文本信息
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (cntrPt[0] + parentPt[0]) / 2.0
    yMid = (cntrPt[1] + parentPt[1]) / 2.0
    createPlot.ax1.text(xMid, yMid, txtString)


# 递归画出决策树
def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    firstStr = myTree.keys()[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


# 创建图像并画出决策树
def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    createPlot.ax1 = plt.subplot(1, 1, 1, frameon=False)
    plotTree.totalW = float(getNumLeafs(inTree))    # 叶子结点总数
    plotTree.totalD = float(getTreeDepth(inTree))   # 树的深度
    plotTree.xOff = - 0.5 / plotTree.totalW
    plotTree.yOff = 1.0 # 初始根结点纵坐标
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()


def savePlot(inTree, filename):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    createPlot.ax1 = plt.subplot(1, 1, 1, frameon=False)
    plotTree.totalW = float(getNumLeafs(inTree))  # 叶子结点总数
    plotTree.totalD = float(getTreeDepth(inTree))  # 树的深度
    plotTree.xOff = - 0.5 / plotTree.totalW
    plotTree.yOff = 1.0  # 初始根结点纵坐标
    plotTree(inTree, (0.5, 1.0), '')
    fig.savefig(filename)


if __name__ == '__main__':
    # createPlot()
    myTree = retrieveTree(0)
    createPlot(myTree)
    savePlot(myTree, 'treeStorage.png')

