import os
import numpy as np
import subprocess

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras import backend
from tensorflow.keras.initializers import RandomNormal

from tensorflow.keras.preprocessing import image_dataset_from_directory


init = RandomNormal(stddev=0.02)

# Define some smaller utility block

# convolutionblock, with normalization and relu
def conv_block(x, size):
   x = layers.Conv2D(size, (4,4), padding='same', kernel_initializer=init)(x)
   x = layers.BatchNormalization()(x)
   x = layers.LeakyReLU(alpha=0.2)(x)
   #x = layers.Conv2D(size, (4,4), padding='same', kernel_initializer=init)(x)
   #x = layers.BatchNormalization()(x)
   #x = layers.LeakyReLU(alpha=0.2)(x)
   return x

# Double image size
def upscale(x):
   img_size = x.shape[1]
   x = tf.image.resize(x, (2*img_size, 2*img_size), method="nearest")
   return x

# Convolution + upscale
def upscale_block(x, size):
   x = conv_block(x, size)
   x = upscale(x)
   return x

# Generate encoded output
def to_rgb(x, n_colors = 3):
   x = layers.Conv2D(n_colors, (4,4), activation='tanh', padding='same', kernel_initializer=init)(x)
   return x

# Upscale with a skip connection
def upscale_skip_block(rgb, x, size):
   x = conv_block(x, size)
   x = upscale(x)
   rgb = upscale(rgb) + to_rgb(x)
   return rgb, x

# convolution + downscale
def downscale_block(x, size):
   x = conv_block(x, size)
   x = layers.Conv2D(size, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(x)
   x = layers.LeakyReLU(alpha=0.2)(x)
   x = layers.BatchNormalization()(x)
   return x

# Downscale with resnet skip
def downscale_block(x, size):
   downscaled = layers.Conv2D(size, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(x)
   x = conv_block(downscaled, size)
   x = x + downscaled
   return x


# generate an intermediate representation from a latent space vector
def make_small_generator(n_in, gcl, n_out):
   input = tf.keras.Input(shape=(n_in))
   n_nodes = gcl * 4 * 4
   x = layers.Dense(n_nodes)(input)
   x = layers.Reshape((4, 4, gcl))(x)
   x = upscale_block(x, gcl)
   x = conv_block(x, gcl)
   output = to_rgb(x, n_out)
   model = Model(inputs = input, outputs = output)
   return model

# generate an image from teh
def make_big_generator(input_shape, gcl, img_size, n_out):
   input = tf.keras.Input(shape=input_shape)
   size = input_shape[1]
   x = input
   rgb = to_rgb(x)
   while size < img_size:
      rgb, x = upscale_skip_block(rgb, x, gcl)
      size *= 2
   model = Model(inputs = input, outputs = rgb)
   return model

# encode the intermediate representations to latent space
def make_small_encoder(input_shape, dcl, latent_dim):
   input = tf.keras.Input(shape=input_shape)
   x = downscale_block(input, dcl)
   x = conv_block(x, dcl)
   x = layers.Flatten()(x)
   x = layers.Dense(latent_dim*2)(x)
   output = layers.Dense(latent_dim)(x)
   model = Model(inputs = input, outputs = output)
   return model

# Encode image to intermediate representation
def make_big_encoder(dcl, img_size, n_out):
   input = tf.keras.Input(shape=(img_size, img_size, 3))
   size = img_size
   s = 1
   x = input
   while size > 8:
      x = downscale_block(x, dcl)
      s*=2
      size /= 2
   x = conv_block(x, dcl)
   output = layers.Conv2D(n_out, (4,4), activation='tanh', padding='same', kernel_initializer=init)(x)
   model = Model(inputs = input, outputs = output)
   return model

# Combine a set of models
def combine_models(steps):
   input_shape = steps[0].input_shape[1:]
   input = tf.keras.Input(shape=input_shape)
   x = input
   for s in steps:
      x =s(x)
   output = x
   model = Model(inputs = input, outputs = output)
   return model



class Autoencoder():
   def __init__(self, IMG_SIZE, size, encoding_size = None,
                learning_rate = 0.0001, beta = 0.9, save_path = None):
      if encoding_size is None:
         self.encoding_size = size
      else:
         self.encoding_size = encoding_size
      self.size = size
      self.IMG_SIZE = IMG_SIZE
      self.encoding_shape = (8, 8, encoding_size)

      if save_path is not None:
         self.save_path = save_path
      else:
         self.save_path = f'autoencoder_{IMG_SIZE}_{size}_{encoding_size}'


      # encoder: image to latent dim
      self.encoder = make_big_encoder(size, IMG_SIZE, encoding_size)
      self.encoder.summary()

      # decoder: latent dim to image
      self.decoder = make_big_generator(self.encoding_shape, size, IMG_SIZE, 3)
      self.decoder.summary()

      # full autoencoder
      self.autoencoder = combine_models((self.encoder, self.decoder))

      self.learning_rate = tf.Variable(learning_rate)
      self.optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate, beta_1=beta)

   @tf.function
   def training_step(self, images):
      with tf.GradientTape(persistent=True) as tape:
         reproduction = self.autoencoder(images)
         loss = tf.math.reduce_mean(tf.math.abs(reproduction - images))

      gradients = tape.gradient(loss, self.autoencoder.trainable_variables)
      self.optimizer.apply_gradients(zip(gradients, self.autoencoder.trainable_variables))
      return loss

   def save(self, bucket = None):
      self.encoder.save(self.save_path+"/encoder")
      self.decoder.save(self.save_path+"/decoder")
      if bucket is not None:
         print("Uploading autoencoder")
         subprocess.call([
          	'gsutil', 'cp', '-r',
      		os.path.join(self.save_path),
          	os.path.join('gs://', bucket)
         ])

   def load(self, bucket = None):
      if bucket is not None:
         print("Downloading autoencoder")
         subprocess.call([
          	'gsutil', 'cp', '-r',
      		os.path.join(self.save_path),
          	os.path.join('gs://', bucket)
         ])
      self.encoder = tf.keras.models.load_model(self.save_path+"/encoder")
      self.decoder = tf.keras.models.load_model(self.save_path+"/decoder")

   def train(self, dataset, epochs, bucket = None, log_step = 50,
             save_every = 5000, learning_rate = 0.0001, lr_update_step = 10000,
             min_learning_rate = 0.00002):
      self.learning_rate.assign(learning_rate)
      n_batches = tf.data.experimental.cardinality(dataset)
      s = 0
      for i in range(epochs):
         for element in dataset:
            s += 1
            j = s%n_batches

            if s%save_every == save_every-1:
               self.save(bucket)
               print("saved")

            if s%lr_update_step == lr_update_step-1:
               if learning_rate > min_learning_rate:
                  learning_rate = learning_rate/2
                  self.learning_rate.assign(learning_rate)

            loss = self.training_step(element)

            if s%log_step == log_step-1:
               print(' %d, %d/%d, ae=%.3f' % (s, j, n_batches, loss))

      print("DONE")
