import tempfile
import urllib
import pandas as pd
import tensorflow as tf
import numpy as np

def input_fn(df):
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.

  continuous_cols = {}
  for k in CONTINUOUS_COLUMNS:
      continuous_cols[k] = tf.constant(df[k].values, name=k)

  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  categorical_cols = {}
  for k in CATEGORICAL_COLUMNS:
    # indices = elements that have nonzero values
    indices = []
    for i in range(df[k].size):
      indices.append([i,0])
    #   print indices
    categorical_cols[k] = tf.SparseTensor(indices=indices, values=df[k].values, shape=[df[k].size,1])
  # categorical_cols = {k: tf.SparseTensor(
  #     indices=[[i, 0] for i in range(df[k].size)],
  #     values=df[k].values,
  #     shape=[df[k].size, 1])
  #                     for k in CATEGORICAL_COLUMNS}
  # Merges the two dictionaries into one.
  feature_cols = dict(continuous_cols.items() + categorical_cols.items())
  # Converts the label column into a constant Tensor.
  label = tf.constant(df[LABEL_COLUMN].values)
  # Returns the feature columns and the label.
  return feature_cols, label

def train_input_fn():
  return input_fn(df_train)

def test_input_fn():
  return input_fn(df_test)


# temporary files for train and test
train_file = tempfile.NamedTemporaryFile()
test_file = tempfile.NamedTemporaryFile()

# get train
urllib.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", train_file.name)
# get test
urllib.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", test_file.name)

# dataset columns
# the last one is the label
COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "gender",
           "capital_gain", "capital_loss", "hours_per_week", "native_country",
           "income_bracket"]

# label column
LABEL_COLUMN = "income_bracket"

# categorical columns
CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation",
                       "relationship", "race", "gender", "native_country"]

# continous columns
CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]

# read dataset
df_train = pd.read_csv(train_file, names=COLUMNS, skipinitialspace=True)
df_test = pd.read_csv(test_file, names=COLUMNS, skipinitialspace=True, skiprows=1)

# remove NaN last element for each column
df_train = df_train.dropna(how='any', axis=0)
df_test = df_test.dropna(how='any', axis=0)

# print df_test['age']
# print len(df_test['age'])
# exit()


# convert data into tensors
train_data, train_label = train_input_fn()
test_data, test_label = test_input_fn()

print train_data
print
print test_data
print
print train_label
print
print test_label
