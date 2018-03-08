import os
from plotly.utils import pandas

import helpers
import pl_stemmer_my

col_used = ['title']
file_name = os.path.dirname(os.path.abspath(__file__)) + "\..\\..\data\\ads\\ads_2017_09_01\\001_anonimized"
df = pandas.read_csv(file_name, header=0, usecols=col_used)
results = dict()

for index, row in df.iterrows():
    stemmed = pl_stemmer_my.stemm_line(row[0])
    for word in stemmed:
        if word in results:
            value = results[word]
            results[word] = value + 1
        else:
            results[word] = 1

with open('stemmed.txt', 'w') as file:
    for key, value in results.iteritems():
        file.write(key.encode('UTF-8'))
        file.write(',')
        file.write(str(value).encode('UTF-8'))
        file.write('\n')