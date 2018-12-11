from sklearn.feature_extration import DictVectorizer
import csv
from sklearn import tree
from sklearn import preprocessing
from sklearn.externals.six import StringIO

with open('sdfsdfsdf.csv', 'rb') as f:
    reader = csv.reader(f)
    headers = reader.next()
    print headers
    featureList = []
    labelList = []

    for row in reader:
        labelList.append(row[len(row) - 1])
        rowDict = {}
        for i in range(1, len(row) - 1):
            rowDict[headers[i]] = row[i]
        featureList.append(rowDict)

    print featureList