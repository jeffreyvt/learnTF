"""
A very simple MNIST classifier
"""

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf


def main(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    x = tf.placeholder(tf.float32, [None, 784])
    W1 = tf.Variable(tf.truncated_normal([784, 30], stddev=0.1))
    W2 = tf.Variable(tf.truncated_normal([30, 10], stddev=0.1))
    b1 = tf.Variable(tf.zeros([30]))
    b2 = tf.Variable(tf.zeros([10]))
    layer1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
    y = tf.matmul(tf.nn.dropout(layer1, 1), W2) + b2
    actual = tf.matmul(layer1, W2) + b2

    y_ = tf.placeholder(tf.float32, [None, 10])

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))+0.00001*(tf.nn.l2_loss(W1)+tf.nn.l2_loss(W2))
    train_step = tf.train.GradientDescentOptimizer(5).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    for _ in range(5000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    correct_prediction = tf.equal(tf.argmax(actual, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
