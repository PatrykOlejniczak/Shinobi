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
import base_predict_helper

col_used = [ce.id, ce.category_id, ce.city_id, ce.user_id, ce.paidads_id_index, ce.has_phone, ce.has_person, ce.predict_sold]
col_used_test = [ce.id, ce.category_id, ce.city_id, ce.user_id, ce.paidads_id_index, ce.has_phone, ce.has_person]

print "Reading CSV files..."
#X_train, Y_train = read_csv(file_name)
X_train, Y_train = base_predict_helper.read_csv_dir(data_dir=base_predict_helper.data_dir, cols=col_used, predict_col_name='predict_sold', first_n_files=10, skip_n_first_files=5)

print "Reading test CSV file..."
#X_test, Y_test = base_predict_helper.read_csv(base_predict_helper.local_test_filename, col_used, predict_col_name='predict_sold')
#X_test, ids = base_predict_helper.read_csv(base_predict_helper.test_filename, col_used_test, predict_col_name='predict_sold', test_file=True)
X_test, ids = base_predict_helper.read_csv(base_predict_helper.file_name, col_used_test, predict_col_name='predict_sold', test_file=True)

print "Testing classifers..."
# selected classifer
# AUC = 0.660174570774 for all ads
# AUC = 0.739155499861 for last three months
#classifer_auc = base_predict_helper.learn_and_test(DecisionTreeClassifier(), X_train, Y_train, X_test, Y_test, "DecisionTreeClassifier")

# other tested classifers
#classifer_auc = base_predict_helper.learn_and_test(GaussianNB(), X_train, Y_train, X_test, Y_test, "GaussianNB", predict_col_name='predict_sold')
#classifer_auc = base_predict_helper.learn_and_test(KNeighborsClassifier(), X_train, Y_train, X_test, Y_test, "KNeighborsClassifier", predict_col_name='predict_sold')
#classifer_auc = base_predict_helper.learn_and_test(LogisticRegression(), X_train, Y_train, X_test, Y_test, "LogisticRegression", predict_col_name='predict_sold')
#classifer_auc = base_predict_helper.learn_and_test(RandomForestClassifier(), X_train, Y_train, X_test, Y_test, "RandomForestClassifier", predict_col_name='predict_sold')
#classifer_auc = base_predict_helper.learn_and_test(AdaBoostClassifier(), X_train, Y_train, X_test, Y_test, "AdaBoostClassifier", predict_col_name='predict_sold')
#classifer_auc = base_predict_helper.learn_and_test(QuadraticDiscriminantAnalysis(), X_train, Y_train, X_test, Y_test, "QuadraticDiscriminantAnalysis", predict_col_name='predict_sold')
#classifer_auc = base_predict_helper.learn_and_test(GaussianProcessClassifier(), X_train, Y_train, X_test, Y_test, "GaussianProcessClassifier", predict_col_name='predict_sold')
#classifer_auc = base_predict_helper.learn_and_test(MLPClassifier(), X_train, Y_train, X_test, Y_test, "MLPClassifier", predict_col_name='predict_sold')

base_predict_helper.learn_and_make_submission(classifer=DecisionTreeClassifier(), X_train=X_train, Y_train=Y_train, X_test=X_test, ids=ids, submission_file_location='..\\..\\textResults\\DecisionTreeClassifier_predict_sold.csv', predict_col_name='predict_sold')
#base_predict_helper.learn_and_make_submission(RandomForestClassifier(), X_train, Y_train, X_test, ids, '..\\..\\textResults\\RandomForestClassifier_predict_sold.csv')