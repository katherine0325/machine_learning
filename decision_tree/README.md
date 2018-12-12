决策树算法

- 有一份文件，是如下数据
```
RID,age,income,student,credit_rating,class_buys_computer
1,youth,high,no,fair,no
2,youth,high,no,excellent,no
3,middle_aged,high,no,fair,yes
4,senior,medium,no,fair,yes
5,senior,low,yes,fair,yes
6,senior,low,yes,excellent,no
7,middle_aged,low,yes,excellent,yes
8,youth,medium,no,fair,no
9,youth,low,yes,fair,yes
10,senior,medium,yes,fair,yes
11,youth,medium,yes,excellent,yes
12,middle_aged,medium,no,excellent,yes
13,middle_aged,high,yes,fair,yes
14,senior,medium,no,excellent,no
```

其中第一列RID是id号，我们不需要使用（即在等下的重组数据格式时将其剔除）
class_buys_computer是最终的结果，即我们标签，应作为结果另外存放

- 通过读取文件数据，将文件数据组成合适的格式
```
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
```

- 将 class_buys_computer的结果存入数组
```
print(labelList)
['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
```

- 将其他的数据存放为json数组
```
print(featureList)
[{
		'age': 'youth',
		'income': 'high',
		'student': 'no',
		'credit_rating': 'fair'
	}, {
		'age': 'youth',
		'income': 'high',
		'student': 'no',
		'credit_rating': 'excellent'
	}, {
		'age': 'middle_aged',
		'income': 'high',
		'student': 'no',
		'credit_rating': 'fair'
	}, {
		'age': 'senior',
		'income': 'medium',
		'student': 'no',
		'credit_rating': 'fair'
	}, {
		'age': 'senior',
		'income': 'low',
		'student': 'yes',
		'credit_rating': 'fair'
	}, {
		'age': 'senior',
		'income': 'low',
		'student': 'yes',
		'credit_rating': 'excellent'
	}, {
		'age': 'middle_aged',
		'income': 'low',
		'student': 'yes',
		'credit_rating': 'excellent'
	}, {
		'age': 'youth',
		'income': 'medium',
		'student': 'no',
		'credit_rating': 'fair'
	}, {
		'age': 'youth',
		'income': 'low',
		'student': 'yes',
		'credit_rating': 'fair'
	}, {
		'age': 'senior',
		'income': 'medium',
		'student': 'yes',
		'credit_rating': 'fair'
	}, {
		'age': 'youth',
		'income': 'medium',
		'student': 'yes',
		'credit_rating': 'excellent'
	}, {
		'age': 'middle_aged',
		'income': 'medium',
		'student': 'no',
		'credit_rating': 'excellent'
	}, {
		'age': 'middle_aged',
		'income': 'high',
		'student': 'yes',
		'credit_rating': 'fair'
	},
	{
		'age': 'senior',
		'income': 'medium',
		'student': 'no',
		'credit_rating': 'excellent'
	}
]
```

- 将特征向量通过 sklearn.feature_extraction DictVectorizer 的 fit_transform 方法转换成 0/1 的数字
```
vec = DictVectorizer()
dummyX = vec.fit_transform(featureList).toarray()
```
    方式：如age这个key的value有 youth, middle_age, senior 三种，那么它就需要被三个 0/1 数字表示，我们可以用 0 0 1 表示 youth, 0 1 0 表示middle_age， 1 0 0 表示 senior，同理其他的key也可以使用这种方法表示，那么我们的特征向量--数组内的第一个json，就可以表示如下：
```
{
    'age': 'youth',
    'income': 'high',
    'student': 'no',
    'credit_rating': 'fair'
}
```
         |         |      |
[0. 0. 1.| 0. 1. 1.| 0. 0.| 1. 0.]
  youth  |   high  |  no  | fair   
         |         |      |

所以结果是：
```
print(dummyX)
[[0. 0. 1. 0. 1. 1. 0. 0. 1. 0.]
 [0. 0. 1. 1. 0. 1. 0. 0. 1. 0.]
 [1. 0. 0. 0. 1. 1. 0. 0. 1. 0.]
 [0. 1. 0. 0. 1. 0. 0. 1. 1. 0.]
 [0. 1. 0. 0. 1. 0. 1. 0. 0. 1.]
 [0. 1. 0. 1. 0. 0. 1. 0. 0. 1.]
 [1. 0. 0. 1. 0. 0. 1. 0. 0. 1.]
 [0. 0. 1. 0. 1. 0. 0. 1. 1. 0.]
 [0. 0. 1. 0. 1. 0. 1. 0. 0. 1.]
 [0. 1. 0. 0. 1. 0. 0. 1. 0. 1.]
 [0. 0. 1. 1. 0. 0. 0. 1. 0. 1.]
 [1. 0. 0. 1. 0. 0. 0. 1. 1. 0.]
 [1. 0. 0. 0. 1. 1. 0. 0. 0. 1.]
 [0. 1. 0. 1. 0. 0. 0. 1. 1. 0.]]
```

- 同样的，结果也要转换为向量矩阵
```
lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
print(dummyY)
[[0]
 [0]
 [1]
 [1]
 [1]
 [0]
 [1]
 [0]
 [1]
 [1]
 [1]
 [1]
 [1]
 [0]]
```

- 使用sklearn的决策树算法训练它们，得到模型clf
```
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(dummyX, dummyY)
```

- 这个model我们也可以把它保存成文件
```
with open('allElectronicInformationGainOri.dot', 'w') as f:
    f = tree.export_graphviz(clf, feature_names=vec.get_feature_names(), out_file=f)
```

- 再进来一个数据，将它转换为矩阵写法，通过模型，我们可以预测它的结果类型
```
predictedY = clf.predict([newRowX])
print("predictedY: " + str(predictedY))

# predictedY: [1]
```

