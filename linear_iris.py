from sklearn import datasets
from sklearn.cross_validation import train_test_split
import tensorflow as tf
import tempfile

# return data (continous features) and labels in a constant tensor object
def input_fn(data, labels, columns):
    continuous_cols = {}
    # convert data
    for i in range(len(columns)):
        # get features i
        values = data[:,i]
        # get feature i name
        column = columns[i]
        # save constant tensor from this column
        continuous_cols[column] = tf.constant(values, name=column)
    # save constant tensor from labels
    labels = tf.constant(labels, name="labels")
    return continuous_cols, labels

def main():
    # iris conlumns
    CONTINUOUS_COLUMNS = ["sepal_length", "sepal_width",
                          "petal_length", "petal_width"]

    # read dataset
    iris = datasets.load_iris()
    # get feature_names
    FN = iris.feature_names
    # get data
    X = iris.data
    # get labels
    Y = iris.target

    # split train and test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    def train_input_fn():
        return input_fn(X_train, y_train, CONTINUOUS_COLUMNS)

    def test_input_fn():
      return input_fn(X_test, y_test, CONTINUOUS_COLUMNS)

    sepal_length = tf.contrib.layers.real_valued_column("sepal_length")
    sepal_width = tf.contrib.layers.real_valued_column("sepal_width")
    petal_length = tf.contrib.layers.real_valued_column("petal_length")
    petal_width = tf.contrib.layers.real_valued_column("petal_width")

    # build model
    model_dir = tempfile.mkdtemp()

    m = tf.contrib.learn.LinearClassifier(feature_columns=[sepal_length, sepal_width, petal_length, petal_width],  model_dir=model_dir)

    m.fit(input_fn=train_input_fn, steps=50)

    results = m.evaluate(input_fn=test_input_fn, steps=1)

    for key in sorted(results):
        print "%s: %s" % (key, results[key])

if __name__ == "__main__":
    main()
