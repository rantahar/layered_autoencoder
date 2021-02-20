import numpy as np
import os
import subprocess

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras import backend
from tensorflow.keras.initializers import RandomNormal

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing import image
import tensorflow_datasets as tfds


def normalize(image, label):
   image = tf.cast(image, tf.float32)
   image = (image / 127.5) - 1
   return image

def flip(image):
   tf.image.random_flip_left_right(image)
   return image

def canonical_size(image, max_image_size=256, max_encoding_size = 64):
   h = image.shape[0]
   w = image.shape[1]
   if h > max_image_size:
      h = 256
      w = w //(h//256)
   if w > max_image_size:
      w = 256
      h = h //(w//256)
   h = h//max_encoding_size
   w = w//max_encoding_size
   h *= max_encoding_size
   w *= max_encoding_size
   image = tf.image.resize(image, (h,w))
   return image

def get_celeba_normalizer(IMG_SIZE):
   def strip_celeba(data):
      img = data['image']
      img = tf.image.resize(img, (IMG_SIZE,IMG_SIZE))
      img = tf.cast(img, tf.float32)
      img = (img / 127.5) - 1
      return img
   return strip_celeba


def load_image(path):
   img = image.load_img(path)
   return image.img_to_array(img)


def dataset_from_folder(path, IMG_SIZE, BATCH_SIZE, bucket = None):

   if bucket is not None:
      print("Downloading data")
      cmd = [
          'gsutil', '-m', 'cp', '-r',
          os.path.join('gs://', bucket, path+'.zip'),
          './'
      ]
      print(subprocess.list2cmdline(cmd))
      subprocess.call(cmd)
      cmd = [
          'unzip', path+'.zip'
      ]
      print(subprocess.list2cmdline(cmd))
      subprocess.call(cmd)


   dataset = image_dataset_from_directory(path, shuffle=True,
      batch_size=BATCH_SIZE, image_size=(IMG_SIZE,IMG_SIZE))

   dataset = dataset.map(normalize).map(flip)
   return dataset


def list_from_folder(path, max_size=256, bucket = None):
      images = []
      for label in os.scandir(path):
         for file in os.scandir(label):
            img = load_image(file)
            img = normalize(img, 1)
            img = canonical_size(img)
            img = tf.convert_to_tensor(img[None,...])
            images.append(img)
      return images



def get_celeba(IMG_SIZE, BATCH_SIZE):
   train_dataset = tfds.load('celeb_a', split='train', batch_size=BATCH_SIZE,
                    shuffle_files=True).map(get_celeba_normalizer(IMG_SIZE))
   valid_dataset = tfds.load('celeb_a', split='test', batch_size=BATCH_SIZE,
                    shuffle_files=True).map(get_celeba_normalizer(IMG_SIZE))
   return train_dataset, valid_dataset
