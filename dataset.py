#  Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""tf.data.Dataset interface to the MNIST dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import shutil
import tempfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
from tensorflow.python.platform import gfile


def read32(bytestream):
  """Read 4 bytes from bytestream as an unsigned 32-bit integer."""
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def check_image_file_header(filename):
  """Validate that filename corresponds to images for the MNIST dataset."""
  with tf.gfile.Open(filename, 'rb') as f:
    magic = read32(f)
    read32(f)  # num_images, unused
    rows = read32(f)
    cols = read32(f)
    if magic != 2051:
      raise ValueError('Invalid magic number %d in MNIST file %s' % (magic,
                                                                     f.name))
    if rows != 28 or cols != 28:
      raise ValueError(
          'Invalid MNIST file %s: Expected 28x28 images, found %dx%d' %
          (f.name, rows, cols))


def check_labels_file_header(filename):
  """Validate that filename corresponds to labels for the MNIST dataset."""
  with tf.gfile.Open(filename, 'rb') as f:
    magic = read32(f)
    read32(f)  # num_items, unused
    if magic != 2049:
      raise ValueError('Invalid magic number %d in MNIST file %s' % (magic,
                                                                     f.name))


def download(directory, filename):
  """Download (and unzip) a file from the MNIST dataset if not already done."""
  filepath = os.path.join(directory, filename)
  if tf.gfile.Exists(filepath):
    return filepath
  if not tf.gfile.Exists(directory):
    tf.gfile.MakeDirs(directory)
  # CVDF mirror of http://yann.lecun.com/exdb/mnist/
  url = 'https://storage.googleapis.com/cvdf-datasets/mnist/' + filename + '.gz'
  _, zipped_filepath = tempfile.mkstemp(suffix='.gz')
  print('Downloading %s to %s' % (url, zipped_filepath))
  urllib.request.urlretrieve(url, zipped_filepath)
  with gzip.open(zipped_filepath, 'rb') as f_in, \
      tf.gfile.Open(filepath, 'wb') as f_out:
    shutil.copyfileobj(f_in, f_out)
  os.remove(zipped_filepath)
  return filepath


def dataset(directory, images_file, labels_file):
  """Download and parse MNIST dataset."""

  images_file = download(directory, images_file)
  labels_file = download(directory, labels_file)

  check_image_file_header(images_file)
  check_labels_file_header(labels_file)

  def decode_image(image):
    # Normalize from [0, 255] to [0.0, 1.0]
    image = tf.decode_raw(image, tf.uint8)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [784])
    return image / 255.0

  def decode_label(label):
    label = tf.decode_raw(label, tf.uint8)  # tf.string -> [tf.uint8]
    label = tf.reshape(label, [])  # label is a scalar
    return tf.to_int32(label)

  images = tf.data.FixedLengthRecordDataset(
      images_file, 28 * 28, header_bytes=16).map(decode_image)
  labels = tf.data.FixedLengthRecordDataset(
      labels_file, 1, header_bytes=8).map(decode_label)
  return tf.data.Dataset.zip((images, labels))


def train(directory):
  """tf.data.Dataset object for MNIST training data."""
  return dataset(directory, 'train-images-idx3-ubyte',
                 'train-labels-idx1-ubyte')


def test(directory):
  """tf.data.Dataset object for MNIST test data."""
  return dataset(directory, 't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte')


class Dataset(object):
    """docstring for Dataset"""
    def __init__(self, images, labels):
        super(Dataset, self).__init__()
        assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming depth == 1)
        assert images.shape[3] == 1
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2])
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    def next_batch(self, size):
        """docstring for next_batch"""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0:
          perm0 = np.arange(self._num_examples)
          np.random.shuffle(perm0)
          self._images = self._images[perm0]
          self._labels = self._labels[perm0]
        # Go to the next epoch
        if start + size > self._num_examples:
          # Finished epoch
          self._epochs_completed += 1
          # Get the rest examples in this epoch
          rest_num_examples = self._num_examples - start
          images_rest_part = self._images[start:self._num_examples]
          labels_rest_part = self._labels[start:self._num_examples]
          # Shuffle the data
          perm = np.arange(self._num_examples)
          np.random.shuffle(perm)
          self._images = self._images[perm]
          self._labels = self._labels[perm]
          # Start next epoch
          start = 0
          self._index_in_epoch = size - rest_num_examples
          end = self._index_in_epoch
          images_new_part = self._images[start:end]
          labels_new_part = self._labels[start:end]
          return np.concatenate(
              (images_rest_part, images_new_part), axis=0), np.concatenate(
                  (labels_rest_part, labels_new_part), axis=0)
        else:
          self._index_in_epoch += size
          end = self._index_in_epoch
          return self._images[start:end], self._labels[start:end]

    @property
    def images(self):
        return self._images
    
    @property
    def labels(self):
        return self._labels

class MnistDataset(object):
    """https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/mnist.py"""
    def download_file(self, filename):
        """docstring for download_file"""
        filepath = os.path.join(self.data_dir, filename)
        if not tf.gfile.Exists(filepath):
            url = 'https://storage.googleapis.com/cvdf-datasets/mnist/' + filename
            _, zipped_filepath = tempfile.mkstemp(suffix='.gz')
            print("Downloading %s to %s." % (url, zipped_filepath))
            urllib.request.urlretrieve(url, zipped_filepath)
            with tf.gfile.Open(zipped_filepath, 'rb') as f_in, tf.gfile.Open(filepath, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            # os.remove(zipped_filepath)
        return filepath

    def extract_images(self, f):
        """docstring for extract_images"""
        print('Extracting', f.name)
        with gzip.GzipFile(fileobj=f) as bytestream:
          if read32(bytestream) != 2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                             (magic, f.name))
          num_images = read32(bytestream)
          rows = read32(bytestream)
          cols = read32(bytestream)
          buf = bytestream.read(rows * cols * num_images)
          data = np.frombuffer(buf, dtype=np.uint8)
          data = data.reshape(num_images, rows, cols, 1)
          return data

    def extract_labels(self, f):
        """docstring for extract_labels"""
        print('Extracting', f.name)
        with gzip.GzipFile(fileobj=f) as bytestream:
          if read32(bytestream) != 2049:
            raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                             (magic, f.name))
          num_items = read32(bytestream)
          buf = bytestream.read(num_items)
          return np.frombuffer(buf, dtype=np.uint8)
        
    def __init__(self, data_dir):
        super(MnistDataset, self).__init__()
        if not tf.gfile.Exists(data_dir):
            tf.gfile.MakeDirs(data_dir)
        self.data_dir = data_dir
        self.train = Dataset(
            self.extract_images(gfile.Open(self.download_file('train-images-idx3-ubyte.gz'), 'rb')), 
            self.extract_labels(gfile.Open(self.download_file('train-labels-idx1-ubyte.gz'), 'rb'))
        )
        self.test = Dataset(
            self.extract_images(gfile.Open(self.download_file('t10k-images-idx3-ubyte.gz'), 'rb')), 
            self.extract_labels(gfile.Open(self.download_file('t10k-labels-idx1-ubyte.gz'), 'rb'))
        )