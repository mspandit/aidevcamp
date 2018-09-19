"""A very simple MNIST classifier.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from dataset import MnistDataset

import tensorflow as tf
import socket

FLAGS = None


class Model(object):
    def __init__(self):
        super(Model, self).__init__()
        # Create the model
        self.x = tf.placeholder(tf.float32, [None, 784])
        with dm.get_device():
            W = tf.Variable(tf.zeros([784, 10]))
            b = tf.Variable(tf.zeros([10]))
            self.y = tf.matmul(self.x, W) + b
        # Define loss and optimizer
        self.y_ = tf.placeholder(tf.int64, [None])
        with dm.get_device():
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
            batch_images, batch_labels = data.next_batch(100)
            session.run(
                self.train_step, feed_dict={
                    self.x: batch_images, 
                    self.y_: batch_labels})

    def test(self, session, data):
        return session.run(
            self.accuracy, feed_dict={
                self.x: data.images,
                self.y_: data.labels})


class DeviceManager(object):
    def __init__(self, nodes=None, me=None):
        super(DeviceManager, self).__init__()
        self.nodes = nodes
        if nodes is not None:
            self.task_index = nodes.index(me)
        self.current_device = 0

    def get_session(self):
        if self.nodes is None:
            sess = tf.InteractiveSession()
            tf.global_variables_initializer().run(session=sess)
            return sess
        else:
            server = tf.train.Server(
                tf.train.ClusterSpec({"worker": self.nodes}), 
                job_name="worker", 
                task_index=self.task_index)
            if 0 != self.task_index:
                server.join()
            else:
                sess = tf.Session(server.target)   # Create a session on the server.
                tf.global_variables_initializer().run(session=sess)
                return sess
    
    def get_device(self):
        if self.nodes is not None:
            device = tf.device("/job:worker/task:%d" % self.current_device)
            self.current_device = (self.current_device + 1) % len(self.nodes)
        else:
            device = tf.device("/job:localhost/replica:0/task:0/device:CPU:0")
        return device
                
def main(_):
  # Import data
  mnist = MnistDataset(FLAGS.data_dir)
  model = Model()
  sess = dm.get_session()
  model.train(sess, mnist.train)

  # Test trained model
  print(model.test(sess, mnist.test))

tf.set_random_seed(10059741)

def node_list(filename):
    return ["%s:3000" % node for node in filename.split("\n")]

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_dir',
      type=str,
      default='/tmp/tensorflow/mnist/input_data',
      help='Directory for storing input data')
  parser.add_argument(
      '--as_node',
      type=str)
  parser.add_argument(
      '--machinefile',
      type=str)
  FLAGS, unparsed = parser.parse_known_args()
  if FLAGS.machinefile is None:
      dm = DeviceManager(["localhost:2222", "localhost:2223"], FLAGS.as_node)
  else:
      dm = DeviceManager(node_list(FLAGS.machinefile), "%s:3000" % socket.gethostname())
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
