# coding: utf-8
import numpy as np
import os
import cv2
# from scipy.misc import imread, imresize
from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from util import *
import tensorflow as tf

def read_images_from():
  images = []
  jpg_files_path = glob("data/train/*.jpg")
  for filename in jpg_files_path:
    im = cv2.imread(filename).astype(np.uint8)
    im = cv2.resize(im, (32, 150))  # .convert("L")  # Convert to greyscale
    # print(type(im))
    # get only images name, not path
    image_name = filename[11:-4]
    images.append([int(image_name), im])

  images = sorted(images, key=lambda image: image[0])

  images_only = [np.asarray(image[1], np.uint8) for image in images]  # Use unint8 or you will be !!!
  images_only = np.array(images_only)

  print(images_only.shape)
  return images_only
    
def read_labels_from():
  print('Reading labels')
  train_labels = pd.read_csv("data/train.csv")['label'].values
  return train_labels

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# images and labels array as input
def convert_to(images, labels, name):
  num_examples = labels.shape[0]
  if images.shape[0] != num_examples:
    raise ValueError("Images size %d does not match label size %d." %
                     (images.shape[0], num_examples))

  filename = os.path.join('data/', name + '.tfrecord')
  print('Writing', filename)
  writer = tf.python_io.TFRecordWriter(filename)
  for index in range(num_examples):
    image_raw = images[index].tostring()
    label_str = labels[index].encode()
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': _bytes_feature(label_str),
        'image_raw': _bytes_feature(image_raw)}))
    writer.write(example.SerializeToString())
  writer.close()

def make_tfrecord():
    images = read_images_from()
    labels = read_labels_from()
    num_training = 80000
    num_validation = 10000
    num_test = 10000

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size = num_test, random_state = 0)
    print('half done!')
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=num_validation, random_state=0)
    convert_to(X_train, y_train, 'train')
    convert_to(X_val, y_val, 'validation')
    convert_to(X_test, y_test, 'test')
    convert_to(images, labels, 'dataset')

def batch_data(dype, num_epoch=1, batchsize=100):
    filename = None
    image_shape = (32,150,3)
    if(dype == 'training'):
        filename = ["data/train.tfrecord"]
        num_epochs = num_epoch
    elif(dype == 'validation'):
        filename = ["data/validation.tfrecord"]
        num_epochs = 1
    elif(dype == 'test'):
        filename = ["data/test.tfrecord"]
        num_epochs = 1
    else:
        raise error
        
    def parse_fn(example_proto):
        features = {"image_raw": tf.FixedLenFeature((), tf.string),
              "label": tf.FixedLenFeature((), tf.string)}
        parsed_features = tf.parse_single_example(example_proto, features)
        image = tf.decode_raw(parsed_features['image_raw'], tf.uint8)
        image = tf.reshape(image, image_shape)
        label = parsed_features['label']
        return image, label

    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(parse_fn)
    dataset = dataset.shuffle(buffer_size=800)
    dataset = dataset.batch(batchsize)
    dataset = dataset.repeat(num_epochs)
    return dataset
