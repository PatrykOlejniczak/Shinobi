from columns import ColumnsEnum as ce
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
import pandas
import os
import time
import datetime
from sklearn import metrics
import csv

data_dir = os.path.dirname(os.path.abspath(__file__)) + "\..\\..\data\\ads"
file_name = os.path.dirname(os.path.abspath(__file__)) + "\..\\..\data\\ads\\ads_2017_09_01\\001_anonimized"
local_test_filename = os.path.dirname(os.path.abspath(__file__)) + "\..\\..\data\\ads\\ads_2017_08_01\\001_anonimized"
test_filename = os.path.dirname(os.path.abspath(__file__)) + "\..\\..\data\\ads_test\\ads_2017_10_01\\ads_2017_10_01"
col_used = [ce.id, ce.category_id, ce.city_id, ce.user_id, ce.paidads_id_index, ce.has_phone, ce.has_person, ce.predict_sold]
col_used_test = [ce.id, ce.category_id, ce.city_id, ce.user_id, ce.paidads_id_index, ce.has_phone, ce.has_person]

def read_csv(csv_file_name, cols, test_file=False):
    X = pandas.read_csv(csv_file_name, header=0, usecols=cols, index_col=None)
    #X = X.dropna()
    X = X.fillna(-1)
    X['has_phone'] = X['has_phone'].map({'t': 1, 'f': 0})
    X['has_person'] = X['has_person'].map({'t': 1, 'f': 0})

    ids = X['id']
    X = X.drop('id', axis=1)

    if test_file:
        return X, ids

    Y = X['predict_sold'].map({'t': 1, 'f': 0})
    X = X.drop('predict_sold', axis=1)

    return X, Y

def read_csv_dir():
    dirs = []
    for path, subdirs, files in os.walk(data_dir):
        for name in files:
            dirs.append(os.path.join(path, name))

    daily_ads = []
    for file_index, current_file_name in enumerate(dirs):
        if file_index < 12:
            if ".git" not in current_file_name:
                print "Reading file {0}".format(current_file_name)
                daily_ads.append(pandas.read_csv(current_file_name, header=0, usecols=col_used))

    X = pandas.concat(daily_ads)
    #X = X.dropna()
    X = X.fillna(-1)
    X['has_phone'] = X['has_phone'].map({'t': 1, 'f': 0})
    X['has_person'] = X['has_person'].map({'t': 1, 'f': 0})

    Y = X['predict_sold'].map({'t': 1, 'f': 0})
    X = X.drop('predict_sold', axis=1)
    X = X.drop('id', axis=1)

    return X, Y

def learn_and_test(classifer, X_train, Y_train, X_test, Y_test, classifer_name, print_log=False):
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
        print "Predicting..."
    Y_predicted = classifer.predict(X_test)

    if print_log:
        print "Calculating AUC..."
    fpr, tpr, thresholds = metrics.roc_curve(Y_test.values.ravel(), Y_predicted)
    auc_score = metrics.auc(fpr, tpr)

    print "AUC = {0} for classifer {1}".format(auc_score, classifer_name)

    return auc_score

def learn_and_make_submission(classifer, X_train, Y_train, X_test, ids, submission_file_location):
    classifer.fit(X_train, Y_train)
    predicted = classifer.predict(X_test)
    predict_proba = classifer.predict_proba(X_test)
    with open(submission_file_location, 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        counter = 0
        for id in ids:
            spamwriter.writerow([id, 0, 0, predict_proba[counter][1]])
            counter += 1


print "Reading CSV files..."
#X_train, Y_train = read_csv(file_name)
X_train, Y_train = read_csv_dir()

print "Reading test CSV file..."
#X_test, Y_test = read_csv(local_test_filename, col_used)
X_test, ids = read_csv(test_filename, col_used_test, True)

print "Testing classifers..."
# selected classifer AUC = 0.660174570774
#classifer_auc = learn_and_test(DecisionTreeClassifier(), X_train, Y_train, X_test, Y_test, "DecisionTreeClassifier")

# other tested classifers
#classifer_auc = learn_and_test(GaussianNB(), X_train, Y_train, X_test, Y_test, "GaussianNB")
#classifer_auc = learn_and_test(KNeighborsClassifier(), X_train, Y_train, X_test, Y_test, "KNeighborsClassifier")
#classifer_auc = learn_and_test(LogisticRegression(), X_train, Y_train, X_test, Y_test, "LogisticRegression")
#classifer_auc = learn_and_test(RandomForestClassifier(), X_train, Y_train, X_test, Y_test, "RandomForestClassifier")
#classifer_auc = learn_and_test(AdaBoostClassifier(), X_train, Y_train, X_test, Y_test, "AdaBoostClassifier")
#classifer_auc = learn_and_test(QuadraticDiscriminantAnalysis(), X_train, Y_train, X_test, Y_test, "QuadraticDiscriminantAnalysis")
#classifer_auc = learn_and_test(GaussianProcessClassifier(), X_train, Y_train, X_test, Y_test, "GaussianProcessClassifier")
#classifer_auc = learn_and_test(MLPClassifier(), X_train, Y_train, X_test, Y_test, "MLPClassifier")

learn_and_make_submission(DecisionTreeClassifier(), X_train, Y_train, X_test, ids, '..\\..\\textResults\\DecisionTreeClassifier.csv')
#learn_and_make_submission(RandomForestClassifier(), X_train, Y_train, X_test, ids, '..\\..\\textResults\\RandomForestClassifier.csv')