from numpy import genfromtxt
import numpy as np
from sklearn import datasets, linear_model

dataPath = r'./test_data.csv'
deliverData = genfromtxt(dataPath, delimiter=',')

print('data')
print(deliverData)

# 讲解一下这个冒号的用法
# 当数据是矩阵的时候，我们要获取矩阵中的一部分
# []里面的第一个位置，代表的是取多少行，:代表全部
# []第二个位置，代表取多少列，使用:全部并倒数第一列用负数表示
X = deliverData[:, :-1]
Y = deliverData[:, -1]

print('X:')
print(X)
print('Y:')
print(Y)

regr = linear_model.LinearRegression()
regr.fit(X, Y)

print('coefficients')
print(regr.coef_)
print('intercept: ')
print(regr.intercept_)

x_pred = [[102, 6]]
y_pred = regr.predict(x_pred)
print('predict y_pred: ')
print(y_pred)