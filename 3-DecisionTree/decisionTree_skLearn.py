# coding=utf-8


from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd


clf = DecisionTreeClassifier(criterion='entropy', max_depth=8)
'''
---
参数：
criterion: 选择最优切分点算法, 'gini'基尼指数, 'entropy'信息增益
max_depth: 树的最大深度
min_samples_split: 一个中间结点至少包含的用例数 
min_samples_leaf: 一个叶子结点至少包含的用例数
max_features: 选择分裂点时考虑的特征数量
'''


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
    data = data.append(data_test)

    # # 把data中的连续值转化为离散值, 可由skLearn自行处理
    # labels_mean_dict = {'age': 37, 'fnlwgt': 1.781445e+05, 'education-num': 10, 'capital-gain': 0, 'capital-loss': 0,
    #                     'hours-per-week': 40}
    # for label in labels:
    #     if type(data[label][0]).__name__ == 'float64':
    #         # print(label)
    #         data[label] = data[label].map(lambda x: str(x > labels_mean_dict[label]))

    # # 仅看非数值列
    # data = data[['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'salary']]
    # labels = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'salary']

    data = data.values

    # LabelEncoder 处理离散化的值
    for col in range(data.shape[1]):
        le = LabelEncoder()
        data[:, col] = le.fit_transform(data[:, col])

    len_train = data.shape[0] - data_test.shape[0]
    X_train = data[:len_train, 0:-1].tolist()
    y_train = data[:len_train, -1].tolist()
    X_test = data[len_train:, 0:-1].tolist()
    y_test = data[len_train:, -1].tolist()

    clf.fit(X_train, y_train)
    predict = clf.predict(X_train)
    print(clf.score(X_train, y_train))
    print(confusion_matrix(y_train, predict))   # 顺序有关, 真实值在前, 预测值在后

    predict = clf.predict(X_test)
    print(accuracy_score(predict, y_test))
    print(confusion_matrix(y_test, predict))

    # treeModel = DecisionTreeClassifier()
    # parameters = {'criterion': ['gini', 'entropy'], 'max_depth': range(1, 10), 'max_features': ['auto', 'sqrt', 'log2', None]}
    # gridSearch = GridSearchCV(treeModel, parameters, n_jobs=1, verbose=5)
    # gridSearch.fit(X_train, y_train)
    # print(gridSearch.best_score_)
    # print(gridSearch.best_params_)
    # bestModel = gridSearch.best_estimator_
    # print(bestModel)
    # predict = bestModel.predict(X_test)
    # print(accuracy_score(predict, y_test))
    # print(confusion_matrix(y_test, predict))

    from sklearn.ensemble import RandomForestClassifier
    forestModel = RandomForestClassifier()
    forestModel.fit(X_train, y_train)
    predict = forestModel.predict(X_train)
    print(confusion_matrix(y_train, predict))
    print(forestModel.score(X_train, y_train))
    print(forestModel)
    print(forestModel.score(X_test, y_test))

    # parameters = {'n_estimators': range(5,20), 'max_depth': range(3, 8)}
    # gridSearch = GridSearchCV(forestModel, parameters, n_jobs=1, verbose=5)
    # gridSearch.fit(X_train, y_train)
    # print(gridSearch.best_score_)
    # print(gridSearch.best_params_)
    # bestModel = gridSearch.best_estimator_
    # print(bestModel)
