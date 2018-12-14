KNN

有以下csv数据(header为经度和纬度)
```
jingdu,weidu,type
1.0,1.1,A
1.0,1.0,A
0,0,B
0,0.1,B
```
以上数据，最后一列为类别

- 将以上需要训练的数据写成数组形式
```
[
    [1.0,1.1],
    [1.0,1.0],
    [0,0],[0,0.1]
]
```

- 同样的，结果也写成数组形式
```
['A','A','B','B']
```

- 然后将KNN的K值设为1
```
knn = KNeighborsClassifier(n_neighbors=1)
```

- 训练它们
```
knn.fit(X_train, Y_train)
```

- 这是有两条新数据，需要我们用knn去预测他们属于什么类别
```
result = knn.predict([[5,0],[4,0]])
print(result)
# ['A' 'A']
```