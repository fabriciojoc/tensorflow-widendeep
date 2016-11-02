# ref: www.tensorflow.org/versions/r0.11/tutorials/wide_and_deep/index.html

import tempfile
import urllib
import pandas as pd
import tensorflow as tf
import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

##
## 1 - READ DATA - movielens 100k
## 100,000 ratings from 1000 users on 1700 movies

# temporary files for train and test
train_file = tempfile.NamedTemporaryFile()
test_file = tempfile.NamedTemporaryFile()

# get data
movielens = fetch_movielens()

# train and test split
# 943 users x 1682 movies
train = movielens['train'].toarray()
test = movielens['test'].toarray()
labels = movielens['item_labels']
print labels[0]
print labels.shape

# transpose to create tensors by movie
train = train.T
test = test.T

# categorical columns
CATEGORICAL_COLUMNS = ['labels']

# continous columns
CONTINUOUS_COLUMNS = [('movie' + str(i)) for i in range(len(labels))]

def to_dict(categorical_data, categorical_columns,
            continuous_data, continuous_columns):
    dictionary = {}
    for d, c in zip(categorical_data, categorical_columns):
        dictionary[c] = d
    for d, c in zip(continuous_data, continuous_columns):
        dictionary[c] = d
    return dictionary

# convert train and test to dict
df_train = to_dict([labels], CATEGORICAL_COLUMNS, train, CONTINUOUS_COLUMNS)
df_test = to_dict([labels], CATEGORICAL_COLUMNS, test, CONTINUOUS_COLUMNS)

##
## 2 - CONVERT DATA TO TENSORS
##

def input_fn(df):
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.

  continuous_cols = {}
  for k in CONTINUOUS_COLUMNS:
      continuous_cols[k] = tf.constant(df[k], name=k)
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  categorical_cols = {}
  for k in CATEGORICAL_COLUMNS:
    # indices = elements that have nonzero values
    indices = []
    for i in range(df[k].size):
      indices.append([i,0])
    categorical_cols[k] = tf.SparseTensor(indices=indices, values=df[k], shape=[df[k].size,1])
  # Merges the two dictionaries into one.
  feature_cols = dict(continuous_cols.items() + categorical_cols.items())
  # Returns feature columns
  return feature_cols

def train_input_fn():
  return input_fn(df_train)

def test_input_fn():
  return input_fn(df_test)

##
## 3 - MODEL FEATURES
##

# categorical feature columns

def sparse_column(n):
    return tf.contrib.layers.sparse_column_with_hash_bucket(n, hash_bucket_size=100)

categorical_columns = [sparse_column(n) for n in CATEGORICAL_COLUMNS]

print categorical_columns

# continuous feature columns
def real_valued_column(n):
    return tf.contrib.layers.real_valued_column(n)

continuous_columns = [real_valued_column(n) for n in CONTINUOUS_COLUMNS]

##
## 4 - WIDE COLUMNS
##

wide_columns = categorical_columns

##
## 5 - DEEP COLUMNS
##
# Each of the sparse, high-dimensional categorical features are first converted
# into a low-dimensional and dense real-valued vector, often referred to as an
# embedding vector


deep_columns = continuous_columns

##
## 6 - MODEL CREATION
##

exit()

model_dir = tempfile.mkdtemp()

m = tf.contrib.learn.DNNLinearCombinedClassifier(
    model_dir=model_dir,
    linear_feature_columns=wide_columns,
    dnn_feature_columns=deep_columns,
    dnn_hidden_units=[100, 50])

##
## 7 - MODEL TRAIN AND TEST
##

m.fit(input_fn=train_input_fn, steps=200)
results = m.evaluate(input_fn=test_input_fn, steps=1)
for key in sorted(results):
    print "%s: %s" % (key, results[key])
