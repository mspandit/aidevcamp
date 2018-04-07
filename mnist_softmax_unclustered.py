"""A very simple MNIST classifier.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


class Model(object):
    def __init__(self):
        super(Model, self).__init__()
        # Create the model
        self.x = tf.placeholder(tf.float32, [None, 784])
        W = tf.Variable(tf.zeros([784, 10]))
        b = tf.Variable(tf.zeros([10]))
        self.y = tf.matmul(self.x, W) + b
        # Define loss and optimizer
        self.y_ = tf.placeholder(tf.int64, [None])
        self.train_step = tf.train.GradientDescentOptimizer(0.5).minimize(self.cross_entropy())
        correct_prediction = tf.equal(tf.argmax(self.y, 1), self.y_)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    def cross_entropy(self):
        """
        The raw formulation of cross-entropy,

          tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
                                        reduction_indices=[1]))

        can be numerically unstable.

        So here we use tf.losses.sparse_softmax_cross_entropy on the raw
        outputs of 'y', and then average across the batch.
        """
        return tf.losses.sparse_softmax_cross_entropy(labels=self.y_, logits=self.y)

    def train(self, session, data):
        for _ in range(1000):
            batch_inputs, batch_targets = data.next_batch(100)
            session.run(
                self.train_step, feed_dict={
                    self.x: batch_inputs, 
                    self.y_: batch_targets})

    def test(self, session, data):
        return session.run(
              self.accuracy, feed_dict={
                  self.x: data.images,
                  self.y_: data.labels})


def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir)
  model = Model()

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  model.train(sess, mnist.train)

  # Test trained model
  print(model.test(sess, mnist.test))

tf.set_random_seed(10059741)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_dir',
      type=str,
      default='/tmp/tensorflow/mnist/input_data',
      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
