from sklearn import datasets
import tensorflow as tf

iris = datasets.load_iris()
FN = iris.feature_names
X = iris.data
Y = iris.target

CONTINUOUS_COLUMNS = ["sepal_length", "sepal_width",
                      "petal_length", "petal_width"]

x = {}
# convert data
for i in range(len(CONTINUOUS_COLUMNS)):
    values = X[:,i]
    column = CONTINUOUS_COLUMNS[i]
    x[column] = tf.constant(values, name=column)

# convert labels
for i in range(len(Y)):
    y = tf.constant(Y, name="labels")
print 'values', x
print 'labels', y
# chars = []
# x = {}
# for i in X:
#     for j in range(len(i)):
#         if not x.has_key(CONTINUOUS_COLUMNS[j]):
#             x[CONTINUOUS_COLUMNS[j]] = []
#         x[CONTINUOUS_COLUMNS[j]].append(i[j])
#
# for c in CONTINUOUS_COLUMNS:
#     print tf.constant(x[c], name=c)


# continuous_cols = {k: tf.constant(new_X[k].values)
#                  for k in CONTINUOUS_COLUMNS}
#
# print continuous_cols
#
# continuous_cols = new_X

# sepal_length = tf.contrib.layers.real_valued_column("sepal_length")
# sepal_width = tf.contrib.layers.real_valued_column("sepal_width")
# petal_length = tf.contrib.layers.real_valued_column("petal_length")
# petal_width = tf.contrib.layers.real_valued_column("petal_width")
#
# print FN
# print X[0]
# print X
