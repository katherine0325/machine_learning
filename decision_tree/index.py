# -*- coding: UTF-8 -*-

from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import tree
from sklearn import preprocessing
from sklearn.externals.six import StringIO

with open(r'data_file.csv', 'rt') as f:
    reader = csv.reader(f)
    headers = next(reader)
    featureList = []
    labelList = []

    for row in reader:
        labelList.append(row[len(row) - 1])
        rowDict = {}
        for i in range(1, len(row) - 1):
            rowDict[headers[i]] = row[i]
        featureList.append(rowDict)

# 将特征变为 sklearn 可输入的形式 -- 向量
vec = DictVectorizer()
dummyX = vec.fit_transform(featureList).toarray()

# sklearn 有专门将二值标签向量化的函数
lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)

# 决策树算法
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(dummyX, dummyY)

# Visualize model
with open('allElectronicInformationGainOri.dot', 'w') as f:
    f = tree.export_graphviz(clf, feature_names=vec.get_feature_names(), out_file=f)

# 进行预测，将第一个样本的第一个属性改变
# # 改变前
oneRowX = dummyX[0, :]
print(str(oneRowX))

# 改变后
newRowX = oneRowX
newRowX[0] = 1
newRowX[2] = 0
print(str(newRowX))

#预测
predictedY = clf.predict([newRowX])
print("predictedY: " + str(predictedY))
