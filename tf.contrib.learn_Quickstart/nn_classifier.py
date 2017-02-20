import tensorflow as tf
import numpy as np

# Import data sets
IRIS_TRAINING = "iris_training.csv"
IRIS_TEST = "iris_test.csv"

# Load data sets
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TRAINING,
    target_dtype=np.int,
    features_dtype=np.float32,
)
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TEST,
    target_dtype=np.int,
    features_dtype=np.float32,
)

# Specify all features have real-value data
feature_columns = [tf.contrib.layers.real_valued_column('', dimension=4)]

# build 3 layer DNN with 10, 20, 10 hidden layers respectively
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=3,
                                            model_dir='/tmp/iris_model')

# fit model
classifier.fit(x=training_set.data, y=training_set.target, steps=2000)

accuracy_score = classifier.evaluate(x=test_set.data, y=test_set.target)["accuracy"]
print('Accuracy: {}'.format(accuracy_score))