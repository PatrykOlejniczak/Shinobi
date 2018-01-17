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
from sklearn.linear_model import LinearRegression
import pandas
import os
import base_predict_helper
import threading

col_used = [ce.id, ce.category_id, ce.city_id, ce.user_id, ce.paidads_id_index, ce.has_phone, ce.has_person, ce.predict_replies]
col_used_test = [ce.id, ce.category_id, ce.city_id, ce.user_id, ce.paidads_id_index, ce.has_phone, ce.has_person]

def test_classifers_t1(X_train, Y_train, X_test, Y_test):
    base_predict_helper.learn_and_test(DecisionTreeClassifier(), X_train, Y_train, X_test, Y_test, "DecisionTreeClassifier",
                                       predict_col_name='predict_replies')
    base_predict_helper.learn_and_test(GaussianNB(), X_train, Y_train, X_test, Y_test, "GaussianNB",
                                       predict_col_name='predict_replies')
    base_predict_helper.learn_and_test(LogisticRegression(), X_train, Y_train, X_test, Y_test, "LogisticRegression",
                                       predict_col_name='predict_replies')
    base_predict_helper.learn_and_test(RandomForestClassifier(), X_train, Y_train, X_test, Y_test, "RandomForestClassifier",
                                       predict_col_name='predict_replies')


def test_classifers_t2(X_train, Y_train, X_test, Y_test):
    base_predict_helper.learn_and_test(QuadraticDiscriminantAnalysis(), X_train, Y_train, X_test, Y_test,
                                       "QuadraticDiscriminantAnalysis", predict_col_name='predict_replies')
    base_predict_helper.learn_and_test(AdaBoostClassifier(), X_train, Y_train, X_test, Y_test, "AdaBoostClassifier",
                                       predict_col_name='predict_replies')

def test_classifers_t3(X_train, Y_train, X_test, Y_test):
    base_predict_helper.learn_and_test(LinearRegression(), X_train, Y_train, X_test, Y_test, "LinearRegression",
                                       predict_col_name='predict_replies')
    base_predict_helper.learn_and_test(MLPClassifier(), X_train, Y_train, X_test, Y_test, "MLPClassifier",
                                       predict_col_name='predict_replies')

print "Reading CSV files..."
#X_train, Y_train = read_csv(file_name)
#X_train, Y_train = base_predict_helper.read_csv_dir(data_dir=base_predict_helper.data_dir, cols=col_used, predict_col_name='predict_replies', first_n_files=12, skip_n_first_files=5)
X_train, Y_train = base_predict_helper.read_csv_dir(data_dir=base_predict_helper.data_dir, cols=col_used, predict_col_name='predict_replies', first_n_files=10, skip_n_first_files=5)

print "Reading test CSV file..."
#X_test, Y_test = base_predict_helper.read_csv(base_predict_helper.local_test_filename, col_used, predict_col_name='predict_replies')
#X_test, ids = base_predict_helper.read_csv(base_predict_helper.test_filename, col_used_test, predict_col_name='predict_replies', test_file=True)
X_test, ids = base_predict_helper.read_csv(base_predict_helper.file_name, col_used_test, predict_col_name='predict_replies', test_file=True)

print "Testing classifers..."
# selected classifer
'''
t1 = threading.Thread(target=test_classifers_t1, args=(X_train, Y_train, X_test, Y_test))
t1.start()
t2 = threading.Thread(target=test_classifers_t2, args=(X_train, Y_train, X_test, Y_test))
t2.start()
t3 = threading.Thread(target=test_classifers_t3, args=(X_train, Y_train, X_test, Y_test))
t3.start()

t1.join()
t2.join()
t3.join()
'''

base_predict_helper.learn_and_make_submission(DecisionTreeClassifier(), X_train, Y_train, X_test, ids, '..\\..\\textResults\\DecisionTreeClassifier_predict_replies.csv', predict_col_name='predict_replies')
#base_predict_helper.learn_and_make_submission(RandomForestClassifier(), X_train, Y_train, X_test, ids, '..\\..\\textResults\\RandomForestClassifier_predict_replies.csv')