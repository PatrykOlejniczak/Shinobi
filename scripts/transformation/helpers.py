import os
import pandas

# col_used - array of indexes or column names
def read_all_files(col_used):
    data_dir = os.path.dirname(os.path.abspath(__file__)) + "\..\..\data\\ads"

    dirs = []
    for path, subdirs, files in os.walk(data_dir):
        for name in files:
            dirs.append(os.path.join(path, name))

    daily_ads = []
    for file_index, file_name in enumerate(dirs):
        if file_index < 11:
            if ".git" not in file_name:
                daily_ads.append(pandas.read_csv(file_name, header=0, usecols=col_used))

    dataFrame = pandas.concat(daily_ads)
    return dataFrame