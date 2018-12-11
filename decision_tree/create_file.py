import csv

headers = ['RID','age','income','student','credit_rating','class_buys_computer']
rows = [('1','youth','high','no','fair','no'),
      ('2','youth','high','no','excellent','no'),
      ('3','middle_aged','high','no','fair','yes'),
      ('4','senior','medium','no','fair','yes'),
      ('5','senior','low','yes','fair','yes'),
      ('6','senior','low','yes','excellent','no'),
      ('7','middle_aged','low','yes','excellent','yes'),
      ('8','youth','medium','no','fair','no'),
      ('9','youth','low','yes','fair','yes'),
      ('10','senior','medium','yes','fair','yes'),
      ('11','youth','medium','yes','excellent','yes'),
      ('12','middle_aged','medium','no','excellent','yes'),
      ('13','middle_aged','high','yes','fair','yes'),
      ('14','senior','medium','no','excellent','no')]
with open('cc.csv', 'wb') as f:
    f_csv = csv.writer(f, dialect = 'excel')
    f_csv.writerow(headers)
    f_csv.writerows(rows)