import data_ninja_loss
import pandas
import numpy
import threading
import traceback

def calc_loss(y_true, y_predicted, set_name):
    try:
        loss = data_ninja_loss._data_ninja_loss(y_true, y_predicted)
        print "Loss for {0}: {1}".format(set_name, loss)
    except:
        print "Calculation failed for {0}".format(set_name)
        traceback.print_exc()

result = pandas.read_csv("..\\textResults\\result.csv", header=None, index_col=None, nrows=10000)
result_matrix = result.as_matrix()

y_sold = []
y_views = []
y_replies = []

counter = 0
for row in result_matrix:
    y_sold.append(row[3])
    y_views.append(row[1])
    y_replies.append(row[2])

test_sold = pandas.read_csv("..\\data\\ads\\ads_2017_09_01\\001_anonimized", header=0, usecols=[21], nrows=10000)
test_replies = pandas.read_csv("..\\data\\ads\\ads_2017_09_01\\001_anonimized", header=0, usecols=[22], nrows=10000)
test_views = pandas.read_csv("..\\data\\ads\\ads_2017_09_01\\001_anonimized", header=0, usecols=[23], nrows=10000)

thread_sold = threading.Thread(target=calc_loss, args=[test_sold.values.ravel(), numpy.asarray(y_sold), 'sold'])
thread_views = threading.Thread(target=calc_loss, args=[test_views.values.ravel(), numpy.asarray(y_views), 'views'])
thread_replies = threading.Thread(target=calc_loss, args=[test_replies.values.ravel(), numpy.asarray(y_replies), 'replies'])

thread_sold.start()
thread_views.start()
thread_replies.start()

thread_sold.join()
thread_views.join()
thread_replies.join()