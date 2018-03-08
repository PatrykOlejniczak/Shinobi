import helpers

def save_uniques(df, column):
    uniques = df[column].unique()

    with open(column, "w") as file:
        for category in uniques:
            file.write(str(category) + "\n")

col_used = ['category_id', 'city_id', 'user_id', 'paidads_id_index']
df = helpers.read_all_files(col_used)

print "min"
print df.min()

print "max"
print df.max()

print "nunique"
print df.nunique()

save_uniques(df, 'category_id')
save_uniques(df, 'paidads_id_index')