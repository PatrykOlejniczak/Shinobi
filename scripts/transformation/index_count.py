import helpers

col_used = ['category_id', 'city_id', 'user_id', 'paidads_id_index']
df = helpers.read_all_files(col_used)

print "min"
print df.min()

print "max"
print df.max()

print "nunique"
print df.nunique()
