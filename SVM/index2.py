# 稍微比较复杂的例子

# numpy 是一个支持矩阵运算的数据包
import numpy as np
# pylab 是一个用于画图的包
import pylab as pl
from sklearn import svm

# 随机抓出某组固定的值
# 比如说，参数填入0，则下面使用随机抓取的时候，无论运行多少次，都是同一批值，如果换成1，则是另一批
np.random.seed(0)
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
Y = [0] * 20 + [1] * 20

clf = svm.SVC(kernel='linear')
clf.fit(X, Y)

# 模型clf创建好之后，我们希望把它画出来，于是有很多值我们要想办法取出来
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - (clf.intercept_[0]) / w[1]

b = clf.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0]) 
b = clf.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])

# 将关键打印出来
print('w: ', w)
print('a: ', a)

print('support_vectors_: ', clf.support_vectors_)
print('clf.coef: ', clf.coef_)

# 画出来
pl.plot(xx, yy, 'k-')
pl.plot(xx, yy_down, 'k--')
pl.plot(xx, yy_up, 'k--')

pl.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80, facecolors='none')
pl.scatter(X[:, 0], X[:, 1], c=Y, cmap=pl.cm.Paired)

pl.axis('tight')
pl.show()



