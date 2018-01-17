import pandas
import csv

result_file = "..\\textResults\\result.csv"
sold_file = "..\\textResults\\DecisionTreeClassifier_predict_sold.csv"
views_file = "..\\textResults\\LinearRegression_predict_views.csv"
replies_file = "..\\textResults\\DecisionTreeClassifier_predict_replies.csv"

sold = pandas.read_csv(sold_file, header=0, index_col=None)
views = pandas.read_csv(views_file, header=0, index_col=None)
replies = pandas.read_csv(replies_file, header=0, index_col=None)

sold_matrix = sold.as_matrix()
views_matrix = views.as_matrix()
replies_matrix = replies.as_matrix()

with open(result_file, 'wb') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

    counter = 0
    for sold_row in sold_matrix:
        id = int(sold_row[0])
        sold_value = sold_row[1]
        views_value = views_matrix[counter][1]
        replies_value = replies_matrix[counter][1]
        csvwriter.writerow([id, views_value, replies_value, sold_value])
        counter += 1