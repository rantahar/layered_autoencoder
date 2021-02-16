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
import tensorflow_datasets as tfds



def normalize(image, label):
   image = tf.cast(image, tf.float32)
   image = (image / 127.5) - 1
   return image

def flip(image):
   tf.image.random_flip_left_right(image)
   return image


def get_celeba_normalizer(IMG_SIZE):
   def strip_celeba(data):
      img = data['image']
      img = tf.image.resize(img, (IMG_SIZE,IMG_SIZE))
      img = tf.cast(img, tf.float32)
      img = (img / 127.5) - 1
      return img
   return strip_celeba


def from_folder(path, IMG_SIZE, BATCH_SIZE, bucket = None):

   if bucket is not None:
      print("Downloading data")
      cmd = [
          'gsutil', '-m', 'cp', '-r',
          os.path.join('gs://', GCP_BUCKET, path+'.zip'),
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

def get_celeba(IMG_SIZE, BATCH_SIZE):
   dataset = tfds.load('celeb_a', split='train', batch_size=BATCH_SIZE,
                    shuffle_files=True).map(get_celeba_normalizer(IMG_SIZE))
   return dataset
