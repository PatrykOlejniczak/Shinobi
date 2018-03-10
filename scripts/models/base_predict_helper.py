import pandas
import os
from sklearn import metrics
import csv
import time
import datetime
import traceback
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import pl_stemmer_my

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion

data_dir = os.path.dirname(os.path.abspath(__file__)) + "\..\\..\data\\ads"
file_name = os.path.dirname(os.path.abspath(__file__)) + "\..\\..\data\\ads\\ads_2017_09_01\\001_anonimized"
local_test_filename = os.path.dirname(os.path.abspath(__file__)) + "\..\\..\data\\ads\\ads_2017_08_01\\001_anonimized"
#test_filename = os.path.dirname(os.path.abspath(__file__)) + "\..\\..\data\\ads_test\\ads_2017_10_01\\ads_2017_10_01"
test_filename = os.path.dirname(os.path.abspath(__file__)) + "\..\\..\data\\ads_test\\ads_2018_03_06\\ads_2018_03_06"
#test_filename = os.path.dirname(os.path.abspath(__file__)) + "\..\\..\data\\bonus_round\\ads_2018_01_10"


def read_csv(csv_file_name, cols, predict_col_name, test_file=False):
    X = pandas.read_csv(csv_file_name, header=0, usecols=cols, index_col=None)
    # X = X.dropna()
    #X = X.fillna(-1)
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
    #X = X.fillna(-1)
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

# source: http://scikit-learn.org/stable/auto_examples/hetero_feature_union.html
class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        result = data_dict.as_matrix(columns=[self.key])
        return result

class ItemSelectorExcept(BaseEstimator, TransformerMixin):
    def __init__(self, keys):
        self.keys = keys

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        local_copy = data_dict.copy(deep=True)
        for key in self.keys:
            local_copy = local_copy.drop(key, axis=1)
        return local_copy

class FillNa(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        nans = np.isnan(data_dict)
        data_dict[nans] = 0
        return data_dict

#source: http://scikit-learn.org/stable/auto_examples/hetero_feature_union.html
class Categories(BaseEstimator, TransformerMixin):
    def __init__(self, file_name):
        self.name = file_name
        self.categories = dict()
        counter = 0
        script_dir = os.path.dirname(__file__)
        with open(script_dir + '\..\..\\textResults\\'+file_name, 'r') as file:
            for category in file:
                if category != '':
                    category = category.replace('\n', '')
                    self.categories[category] = counter
                    counter += 1

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        result = np.zeros(X.shape)
        counter = 0
        for row in X:
            try:
                result[counter][self.categories[str(row[0])]] = 1
            except:
                pass
            counter += 1

        return result

class UsedWords(BaseEstimator, TransformerMixin):
    def __init__(self, dict_size):
        self.dict_size = dict_size
        self.words = dict()
        counter = 0
        script_dir = os.path.dirname(__file__)
        with open(script_dir + '\..\..\\textResults\\stemmed.txt', 'r') as file:
            for line in file:
                word, value = line.split(',')
                self.words[word] = counter
                counter += 1
                if counter > self.dict_size:
                    break

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        result = []
        print "Stemming started"
        counter = 0
        for row in X:
            line = np.zeros((self.dict_size,), dtype=int)
            stemmed_title = pl_stemmer_my.stemm_line(row[0])

            for stemmed_word in stemmed_title:
                try:
                    line[self.words[stemmed_word]] = 1
                except:
                    pass

            result.append(line)
            counter += 1
            if counter % 1000 == 0:
                print str(counter)

        return np.asarray(result)

#source: http://scikit-learn.org/stable/auto_examples/hetero_feature_union.html
def prepare_pipeline(classifier, classifier_name):
    pipeline = Pipeline([
        # Use FeatureUnion to combine the features
        ('union', FeatureUnion(
            transformer_list=[
                ('rest', ItemSelectorExcept(keys=['paidads_id_index', 'title', 'accurate_location'])),

                ('category_id', Pipeline([
                    ('selector', ItemSelector(key='category_id')),
                    ('stats', Categories('category_id'))
                ])),
                ('paidads_id_index', Pipeline([
                    ('selector', ItemSelector(key='paidads_id_index')),
                    ('stats', Categories('paidads_id_index'))
                ])),
                ('accurate_location', Pipeline([
                    ('selector', ItemSelector(key='accurate_location')),
                    ('stats', FillNa())
                ])),
                #('title', Pipeline([
                 #   ('selector', ItemSelector(key='title')),
                  #  ('stats', UsedWords(50))
                #]))
            ]
        )),

        # classifier
        (classifier_name, classifier),
    ])

    return pipeline

def learn_and_make_submission(classifer, X_train, Y_train, X_test, ids, submission_file_location, predict_col_name, classifer_name='classifier'):
    pipeline = prepare_pipeline(classifer, classifer_name)
    #pipeline = classifer
    pipeline.fit(X_train, Y_train)
    predicted = pipeline.predict(X_test)
    predict_proba = []
    if predict_col_name == 'predict_sold':
        predict_proba = pipeline.predict_proba(X_test)
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
