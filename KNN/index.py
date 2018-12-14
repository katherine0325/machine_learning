# -*- coding: UTF-8 -*-

import numpy as np
import sklearn
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

X_train = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
Y_train = ['A','A','B','B']

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, Y_train)
result = knn.predict([[5,0],[4,0]])
print(result)