import pandas as pd  # work with data as tables
import numpy as np  # use number matrices
import matplotlib.pyplot as plt
import tensorflow as tf

# Step 1: Load data
dataframe = pd.read_csv('data.csv')  # dataframe object

# Remove data that we aren't interested in
dataframe = dataframe.drop(['index', 'price', 'sq_price'], axis=1)

# We only use the first 10 rows of data
dataframe = dataframe[0:10]
print(dataframe)

# Step 2: add labels - make the problem into a classification problem
# 1 is good buy and 0 is bad buy
dataframe.loc[:, ("y1")] = [1, 1, 1, 0, 0, 1, 0, 1, 1, 1]
# y2 is the negation of y1
dataframe.loc[:, ("y2")] = dataframe["y1"] == 0
# Turn true or false values into int
dataframe.loc[:, ("y2")] = dataframe["y2"].astype(int)
print(dataframe)

# Step 3: prepare data for tensorflow
# tensors are a generic version of vectors and matrices
# vector is a list of numbers (1D tensor)
# matrix is a list of list of numbers (2D tensor)
# list of list of list numbers (3D tensor)
# ......
# convert features to input tensors
inputX = dataframe.loc[:, ['area', 'bathrooms']].as_matrix()
print(inputX)
# convert labels to input tensors
inputY = dataframe.loc[:, ['y1', 'y2']].as_matrix()
print(inputY)


# Step 4: write out our hyper parameters

learning_rate = 0.000001
training_epochs = 2000
display_steps = 50
n_samples = inputX.size

# Step 5: create our computation graph/neural network
# for feature input tensors, None means any numbers of examples
# placeholders are gateway for data into our computation graph
x = tf.placeholder(tf.float32, [None, 2])

# create weights
# 2x2 float matrix, that will be kept updating throughout the training process
# variables in tf hold and update parameters
# in memory buffers containing tensors
w = tf.Variable(tf.zeros([2, 2]))

# add biases
b = tf.Variable(tf.zeros([2]))

# multiply our weights by our inputs, first calculation
# weights are how we govern how the data flow in our computation graph
# multiply input by weights and add biases
y_value = tf.add(tf.matmul(x, w), b)

# apply softmax function to value which we just created
# softmax is our activation function
y = tf.nn.softmax(y_value)

# feed in a matrix of labels <- actual values
y_ = tf.placeholder(tf.float32, [None, 2])


# Step 6: perform training
# create our cost function, mean square error
# reduce sum computes the sum of elements across dimensions of a tensor
cost = tf.reduce_sum(tf.pow(y_-y, 2)/(2*n_samples))

# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


# initialize variables and tensorflow session
init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init)

# training loop
for i in range(training_epochs):
    sess.run(optimizer, feed_dict={x: inputX, y_: inputY})


    if i % display_steps == 0:
        cc = sess.run(cost, feed_dict={x: inputX, y_: inputY})
        print("Training step: ", i, "cost = ", cc)
print("optimisation finished")
training_cost = sess.run(cost, feed_dict={x: inputX, y_: inputY})
print("training cost = ", training_cost, "\nw=", sess.run(w), "\nb=", sess.run(b))


print(sess.run(y, feed_dict={x:inputX}))

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: inputX, y_: inputY}))