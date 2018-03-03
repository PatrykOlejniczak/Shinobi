import pandas
import os
from sklearn import metrics
import csv
import time
import datetime
import traceback

data_dir = os.path.dirname(os.path.abspath(__file__)) + "\..\\..\data\\ads"
file_name = os.path.dirname(os.path.abspath(__file__)) + "\..\\..\data\\ads\\ads_2017_09_01\\001_anonimized"
local_test_filename = os.path.dirname(os.path.abspath(__file__)) + "\..\\..\data\\ads\\ads_2017_08_01\\001_anonimized"
test_filename = os.path.dirname(os.path.abspath(__file__)) + "\..\\..\data\\ads_test\\ads_2017_10_01\\ads_2017_10_01"
#test_filename = os.path.dirname(os.path.abspath(__file__)) + "\..\\..\data\\bonus_round\\ads_2018_01_10"


def read_csv(csv_file_name, cols, predict_col_name, test_file=False):
    X = pandas.read_csv(csv_file_name, header=0, usecols=cols, index_col=None)
    # X = X.dropna()
    X = X.fillna(-1)
    X['has_phone'] = X['has_phone'].map({'t': 1, 'f': 0})
    X['has_person'] = X['has_person'].map({'t': 1, 'f': 0})

    ids = X['id']
    X = X.drop('id', axis=1)

    if test_file:
        return X, ids

    if predict_col_name == 'predict_sold':
        Y = X['predict_sold'].map({'t': 1, 'f': 0})
    else:
        Y = X[predict_col_name]

    X = X.drop(predict_col_name, axis=1)

    return X, Y


def read_csv_dir(data_dir, cols, predict_col_name, first_n_files=11, skip_n_first_files=0):
    dirs = []
    for path, subdirs, files in os.walk(data_dir):
        for name in files:
            dirs.append(os.path.join(path, name))

    daily_ads = []
    for file_index, current_file_name in enumerate(dirs):
        if first_n_files >= file_index > skip_n_first_files:
            if ".git" not in current_file_name:
                print "Reading file {0}".format(current_file_name)
                daily_ads.append(pandas.read_csv(current_file_name, header=0, usecols=cols))

    X = pandas.concat(daily_ads)
    # X = X.dropna()
    X = X.fillna(-1)
    X['has_phone'] = X['has_phone'].map({'t': 1, 'f': 0})
    X['has_person'] = X['has_person'].map({'t': 1, 'f': 0})

    if predict_col_name == 'predict_sold':
        Y = X['predict_sold'].map({'t': 1, 'f': 0})
    else:
        Y = X[predict_col_name]
    X = X.drop(predict_col_name, axis=1)
    X = X.drop('id', axis=1)

    return X, Y


def learn_and_test(classifer, X_train, Y_train, X_test, Y_test, classifer_name, predict_col_name, print_log=False):
    try:
        if print_log:
            start_time = time.time()
            readable_time = datetime.datetime.fromtimestamp(start_time)
            print "{0}: Learning started at {1}".format(classifer_name, readable_time.strftime('%Y-%m-%d %H:%M:%S'))

        classifer.fit(X_train, Y_train.values.ravel())

        if print_log:
            stop_time = time.time()
            readable_time = datetime.datetime.fromtimestamp(stop_time)
            print "{0}: Learning finished at {1}".format(classifer_name, readable_time.strftime('%Y-%m-%d %H:%M:%S'))

        if print_log:
            print "Calculating score..."
        if predict_col_name == 'predict_sold':
            Y_predicted = classifer.predict(X_test)
            fpr, tpr, thresholds = metrics.roc_curve(Y_test.values.ravel(), Y_predicted)
            auc_score = metrics.auc(fpr, tpr)

            print "AUC = {0} for classifer {1} while predicting {2}".format(auc_score, classifer_name, predict_col_name)

            return auc_score
        else:
            score = classifer.score(X_test, Y_test.values.ravel())
            print "Score = {0} for classifer {1} while predicting {2}".format(score, classifer_name, predict_col_name)
            return score
    except:
        print "Test failure for {0}".format(classifer_name)
        traceback.print_exc()
        return -1


def learn_and_make_submission(classifer, X_train, Y_train, X_test, ids, submission_file_location, predict_col_name):
    classifer.fit(X_train, Y_train)
    predicted = classifer.predict(X_test)
    predict_proba = []
    if predict_col_name == 'predict_sold':
        predict_proba = classifer.predict_proba(X_test)
    with open(submission_file_location, 'wb') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(['id', predict_col_name])

        counter = 0
        for id in ids:
            if predict_col_name == 'predict_sold':
                csvwriter.writerow([id, predict_proba[counter][1]])
            else:
                csvwriter.writerow([id, predicted[counter]])
            counter += 1
