# -*- coding: utf-8 -*-
# Histogramy (łącznie z wykresem) liczby wyświetleń, odpowiedzi oraz sprzedaży (*),

import pandas
import os
import math
import matplotlib.pyplot as plot
from collections import defaultdict

# ["id",
# "region_id",
# "category_id",
# "subregion_id",
# "district_id",
# "city_id",
# "accurate_location",
# "user_id",
# "sorting_date",
# "created_at_first",
# "valid_to",
# "title",
# "description",
# "full_description",
# "has_phone",
# "params", "has_person",
# "photo_sizes",
# "paidads_id_index",
# "paidads_valid_to",
# "predict_sold",
# "predict_replies",
# "predict_views",
# "reply_call",
# "reply_sms",
# "reply_chat",
# "reply_call_intent",
# "reply_chat_intent"]

data_dir = os.path.dirname(os.path.abspath(__file__)) + "\..\data\\ads"
col_used = [0, 21, 22, 23]

dirs = []
for path, subdirs, files in os.walk(data_dir):
    for name in files:
        dirs.append(os.path.join(path, name))

def convertBoolean(a):
    if a == 't':
        return 1
    return 0

daily_ads = []
dataFrame = []
converters = {21: convertBoolean}
for file_index, file_name in enumerate(dirs):
    if file_index < 11:
        if ".git" not in file_name:
            daily_ads.append(pandas.read_csv(file_name, header=0, usecols=col_used, converters=converters))
            #daily_ads.append(pandas.read_csv(file_name, header=0, usecols=col_used, nrows=5, converters=converters))

#print(daily_ads)

#print "\nconcated\n"
dataFrame = pandas.concat(daily_ads)
#print dataFrame

#print "\ngroupby\n"
#groupedDF = dataFrame.loc(dataFrame['predict_sold'] == 'f')
#groupedDF = dataFrame[dataFrame.predict_sold == 't']
#print groupedDF
groupedDF = dataFrame.groupby(['id']).sum()
groupedDF = groupedDF[groupedDF.predict_sold > 0]
print groupedDF

column = "predict_views"
maxValue = groupedDF[column].max()
step = 500
#step = 10
#groupedDF = groupedDF.groupby(pandas.cut(groupedDF["predict_views"], np.arange(0, maxValue+step, step))).count()

#print "\ncut\n"
if math.isnan(maxValue):
    maxValue = 0
print "Max " + "column" + " value: " + str(maxValue)

plot.hist(groupedDF[column], range(0, maxValue+step, step), log=True)
plot.xlabel('Views')
plot.ylabel('Ads - sold')
plot.grid(True)
plot.show()