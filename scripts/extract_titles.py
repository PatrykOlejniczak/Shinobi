import numpy as np
import pandas
import os

file_name = os.path.dirname(os.path.abspath(__file__)) + "\..\data\\ads\\ads_2017_09_01\\001_anonimized"
extracted_file_name = os.path.dirname(os.path.abspath(__file__)) + "\..\data\\ads\\ads_2017_09_01\\001_anonimized_ext"
col_used = [11]

title = pandas.read_csv(file_name, header=0, usecols=col_used, index_col=None)

print title

title.to_csv(extracted_file_name, index=None)