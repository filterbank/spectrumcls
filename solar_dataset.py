# 'solarset.py' is used to read solar dataset,which is derived from 'data4'
#  handled by 'solar_data.py'
# ==============================================================================

"""Functions for reading solar data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.framework import dtypes
import os
import numpy

def _read32(bytestream):
  dt = numpy.dtype(numpy.uint32).newbyteorder('<')
  return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]

def extract_images(filename):
  """Extract the images into a 3D uint8 numpy array [index, y, x]."""
  print('Extracting', filename)
  with open(filename, 'r') as  bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError('Invalid magic number %d in solar image file: %s' %
                       (magic, filename))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = numpy.frombuffer(buf, dtype=numpy.uint8)
    data = data.reshape(num_images, rows, cols)
    return data


def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


def extract_labels(filename, num_classes=3):
  """Extract the labels into a 1D uint8 numpy array [index]."""
  print('Extracting', filename)
  with open(filename, 'r') as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError('Invalid magic number %d in solar label file: %s' %(magic, filename))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8)
    return dense_to_one_hot(labels, num_classes)


class DataSet(object):

  def __init__(self, images, labels, dtype=dtypes.float32, reshape=True):
    """
    Construct a DataSet.
    `dtype` can be either`uint8` to leave the input as `[0, 255]`, or `float32` to rescale into`[0, 1]`.
    """
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
        raise TypeError('Invalid image dtype %r, expected uint8 or float32' % dtype)
    if images.shape[0] != labels.shape[0]:
        raise TypeError('Images.shape: %s labels.shape: %s' % images.shape, labels.shape)
    self._num_examples = images.shape[0]

    # Convert shape from [num examples, rows, columns]
    # to [num examples, rows*columns]
    if reshape:
        images = images.reshape(images.shape[0],images.shape[1] * images.shape[2])
    if dtype == dtypes.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(numpy.float32)
        images = numpy.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]


def read_data_sets(datadirctory,filenamelist,dtype=dtypes.float32,reshape=True):
  #'datadirctory': dirctory of test data ,train lable, train data and test lable,they are in the same dirctory
  #'filenamelist'is a list of file names,their order is traindata,trainlable,testdata,testlable
  if not os.path.exists(datadirctory):
      raise ValueError('The directory of %s is not exists ' % datadirctory)
  filepathlist=[]
  for filename in filenamelist:
      filepath = os.path.join(datadirctory, filename)
      if not os.path.isfile(filepath):
          raise ValueError('The file of %s is not exists' % filepath)
      filepathlist.append(filepath)
  train_images = extract_images(filepathlist[0])
  train_labels = extract_labels(filepathlist[1])

  test_images = extract_images(filepathlist[2])
  test_labels = extract_labels(filepathlist[3])

  train = DataSet(train_images, train_labels, dtype=dtype, reshape=reshape)
  test = DataSet(test_images, test_labels, dtype=dtype, reshape=reshape)

  return train,test


