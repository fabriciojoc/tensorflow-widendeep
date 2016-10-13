import tempfile
import urllib
import pandas as pd
import tensorflow as tf
import numpy as np

##
## 1 - READ DATA
##

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

# read dataset
df_train = pd.read_csv(train_file, names=COLUMNS, skipinitialspace=True)
df_test = pd.read_csv(test_file, names=COLUMNS, skipinitialspace=True, skiprows=1)

# remove NaN last element for each column
df_train = df_train.dropna(how='any', axis=0)
df_test = df_test.dropna(how='any', axis=0)

##
## 2 - CONVERT DATA TO TENSORS
##

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
    categorical_cols[k] = tf.SparseTensor(indices=indices, values=df[k].values, shape=[df[k].size,1])
  # Merges the two dictionaries into one.
  feature_cols = dict(continuous_cols.items() + categorical_cols.items())
  # Convert labels to integer
  labels = []
  uniq_labels = np.unique(df[LABEL_COLUMN].values)
  for i in df[LABEL_COLUMN].values:
      for j in range(len(uniq_labels)):
          if i == uniq_labels[j]:
              labels.append(j)
  # Converts the label column into a constant Tensor.
  label = tf.constant(labels)
  # Returns the feature columns and the label.
  return feature_cols, label

def train_input_fn():
  return input_fn(df_train)

def test_input_fn():
  return input_fn(df_test)

# label column
LABEL_COLUMN = "income_bracket"

# categorical columns
CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation",
                       "relationship", "race", "gender", "native_country"]

# continous columns
CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]

##
## 3 - MODEL FEATURES
##

# categorical feature columns
workclass = tf.contrib.layers.sparse_column_with_hash_bucket("workclass", hash_bucket_size=100)
education = tf.contrib.layers.sparse_column_with_hash_bucket("education", hash_bucket_size=1000)
marital_status = tf.contrib.layers.sparse_column_with_hash_bucket("marital_status", hash_bucket_size=100)
occupation = tf.contrib.layers.sparse_column_with_hash_bucket("occupation", hash_bucket_size=1000)
relationship = tf.contrib.layers.sparse_column_with_hash_bucket("relationship", hash_bucket_size=100)
race = tf.contrib.layers.sparse_column_with_keys(column_name="race", keys=[
  "Amer-Indian-Eskimo", "Asian-Pac-Islander", "Black", "Other", "White"])
gender = tf.contrib.layers.sparse_column_with_keys(
  column_name="gender", keys=["female", "male"])
native_country = tf.contrib.layers.sparse_column_with_hash_bucket("native_country", hash_bucket_size=1000)

# continuous feature columns
age = tf.contrib.layers.real_valued_column("age")
# transform age to categorical
age_buckets = tf.contrib.layers.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
education_num = tf.contrib.layers.real_valued_column("education_num")
capital_gain = tf.contrib.layers.real_valued_column("capital_gain")
capital_loss = tf.contrib.layers.real_valued_column("capital_loss")
hours_per_week = tf.contrib.layers.real_valued_column("hours_per_week")

# crossed features columns
education_x_occupation = tf.contrib.layers.crossed_column([education, occupation], hash_bucket_size=int(1e4))
age_buckets_x_race_x_occupation = tf.contrib.layers.crossed_column(
  [age_buckets, race, occupation], hash_bucket_size=int(1e6))

##
## 4 - MODEL CREATION
##

model_dir = tempfile.mkdtemp()
m = tf.contrib.learn.LinearClassifier(feature_columns=[
  gender, native_country, education, occupation, workclass, marital_status, race, age_buckets, education_x_occupation, age_buckets_x_race_x_occupation],
  optimizer=tf.train.FtrlOptimizer(
    learning_rate=0.1,
    l1_regularization_strength=1.0,
    l2_regularization_strength=1.0),
  model_dir=model_dir)

##
## 5 - MODEL TRAIN AND TEST
##

m.fit(input_fn=train_input_fn, steps=200)
results = m.evaluate(input_fn=test_input_fn, steps=1)
for key in sorted(results):
    print "%s: %s" % (key, results[key])
