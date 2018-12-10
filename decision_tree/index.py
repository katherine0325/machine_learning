#from sklearn.feature_extraction import DictVectorizer
import csv
#from sklearn import preprocessing
#from sklearn import tree
#from sklearn.externals.six import StringIO

allElectronicsData = open(r'F:\internet\machine_learning\decision_tree\test.txt', 'rt')
reader = csv.reader(allElectronicsData)
headers = next(reader)
second = next(reader)
c = next(reader)
d = next(reader)
f = next(reader)

print(headers)
print(second)
print(c)
print(d)
print(f)


