# Histogramy (łącznie z wykresem) liczby wyświetleń, odpowiedzi oraz sprzedaży (*),

import pandas
import os
import matplotlib.pyplot as plot
from collections import defaultdict

#["id", 
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
col_used = [0,21,22,23]

dirs = []
for path, subdirs, files in os.walk(data_dir):
    for name in files:
        dirs.append(os.path.join(path, name))

daily_ads = []
for file_index, file_name in enumerate(dirs):
    if file_index < 3:
        if ".git" not in file_name:
            daily_ads.append(pandas.read_csv(file_name, header = 0, usecols = col_used))

print(daily_ads)