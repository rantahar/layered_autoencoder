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
   x = layers.Conv2D(size, (4,4), padding='same', kernel_initializer=init)(x)
   x = layers.BatchNormalization()(x)
   x = layers.LeakyReLU(alpha=0.2)(x)
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
def upscale_skip_block(rgb, x, size, colors = 3):
   x = conv_block(x, size)
   x = upscale(x)
   rgb = upscale(rgb) + to_rgb(x, colors)
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


# Reduce the image size by a factor of 4 and encode each pixel to n_out numbers
def make_encoder(shape, dcl, n_out):
   input = tf.keras.Input(shape=shape)
   x = downscale_block(input, dcl)
   x = downscale_block(x, dcl)
   output = layers.Conv2D(n_out, (4,4), activation='tanh', padding='same', kernel_initializer=init)(x)
   model = Model(inputs = input, outputs = output)
   return model


# Encode a small image to a single latent space vector
def make_encoder_head(input_shape, dcl, latent_dim):
   input = tf.keras.Input(shape=input_shape)
   size = input_shape[2]
   if size > 4:
      x = downscale_block(input, dcl)
   else:
      x = input
   x = layers.Flatten()(x)
   x = layers.Dense(latent_dim*2)(x)
   output = layers.Dense(latent_dim)(x)
   model = Model(inputs = input, outputs = output)
   return model


# generate small image from a latent space vector
def make_decoder_head(latent_dim, gcl, out_shape):
   input = tf.keras.Input(shape=(latent_dim))
   img_size = out_shape[1]
   if img_size > 4:
      s = img_size // 2
   else:
      s = img_size
   n_nodes = gcl * s * s
   x = layers.Dense(gcl)(input)
   x = layers.Dense(n_nodes)(x)
   x = layers.Reshape((s, s, gcl))(x)
   if img_size > 4:
      x = upscale_block(x, gcl)
   output = to_rgb(x, out_shape[-1])
   model = Model(inputs = input, outputs = output)
   return model

# Increase image size by a factor of 4 and encode each pixel to n_out numbers
def make_decoder(input_shape, gcl, n_out):
   input = tf.keras.Input(shape=input_shape)
   rgb = to_rgb(input, n_out)
   rgb, x = upscale_skip_block(rgb, input, gcl, n_out)
   rgb, x = upscale_skip_block(rgb, x, gcl, n_out)
   model = Model(inputs = input, outputs = rgb)
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
   def __init__(self, IMG_SIZE = 64, size = 32, latent_dim = None,
                encoding_size = None, learning_rate = 0.0001, beta = 0.9,
                save_path = None, load = False):
      if encoding_size is None:
         self.encoding_size = size
      else:
         self.encoding_size = encoding_size

      if latent_dim is None:
         self.latent_dim = size
      else:
         self.latent_dim = latent_dim

      self.size = size
      self.IMG_SIZE = IMG_SIZE

      if save_path is not None:
         self.save_path = save_path
      else:
         self.save_path = f'autoencoder_{IMG_SIZE}_{size}_{self.encoding_size}'

      self.levels = []

      if load:
         self.load()
      else:
         in_shape = (IMG_SIZE, IMG_SIZE, 3)
         while in_shape[1] > 8:

            encoder = make_encoder(in_shape, size, self.encoding_size)
            encoder.summary()

            shape = encoder.output_shape[1:]

            decoder = make_decoder(shape, size, in_shape[-1])
            decoder.summary()

            # full autoencoder
            autoencoder = combine_models((encoder, decoder))

            self.levels.append((encoder, decoder, autoencoder))

            in_shape = shape

         encoder = make_encoder_head(in_shape, size, self.latent_dim)
         encoder.summary()
         decoder = make_decoder_head(latent_dim, size, in_shape)
         decoder.summary()
         autoencoder = combine_models((encoder, decoder))

         self.levels.append((encoder, decoder, autoencoder))

      self.n_levels = len(self.levels)

      self.learning_rate = tf.Variable(learning_rate)
      self.optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate, beta_1=beta)


   def training_step(self, images, level):
      target = images
      for i in range(level):
         target = self.levels[i][0](target)
      print(target.shape)
      autoencoder = self.levels[level][2]
      print(autoencoder.input_shape)
      print(autoencoder.output_shape)
      with tf.GradientTape(persistent=True) as tape:
         reproduction = autoencoder(target)
         loss = tf.math.reduce_mean(tf.math.abs(reproduction - target))

      gradients = tape.gradient(loss, autoencoder.trainable_variables)
      self.optimizer.apply_gradients(zip(gradients, autoencoder.trainable_variables))
      return loss

   def save(self, bucket = None):
      for i, l in enumerate(self.levels):
         l[0].save(self.save_path+f"/encoder{i}")
         l[1].save(self.save_path+f"/decoder{i}")
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
      self.levels = []
      i = 0
      while os.path.isdir(f"{self.save_path}/encoder{i}"):
         encoder = tf.keras.models.load_model(f"{self.save_path}/encoder{i}")
         decoder = tf.keras.models.load_model(f"{self.save_path}/decoder{i}")
         autoencoder = combine_models((encoder, decoder))
         self.levels.append((encoder, decoder, autoencoder))
         i += 1

   @tf.function
   def evaluate(self, images, level):
      x = images
      for i in range(level):
         x = self.levels[i][0](x)
      for i in range(level):
         x = self.levels[level-i-1][1](x)
      loss = tf.math.reduce_mean(tf.math.abs(x - images))
      return loss

   @tf.function
   def generate(self, images, level):
      x = images
      for i in range(level):
         x = self.levels[i][0](x)
      for i in range(level):
         x = self.levels[level-i-1][1](x)
      return x

   def train(self, dataset, epochs, bucket = None, level_target = 0.3, log_step = 50,
             save_every = 5000, learning_rate = 0.0001, lr_update_step = 10000,
             min_learning_rate = 0.00002, level = None):

      self.learning_rate.assign(learning_rate)
      n_batches = tf.data.experimental.cardinality(dataset)
      for level in range(self.n_levels):
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

               loss = self.training_step(element, level)

               if s%log_step == log_step-1:
                  full_loss = self.evaluate(element, level+1)
                  print(' %d, %d/%d, ae=%.3f, full_loss=%.3f' %
                     (s, j, n_batches, loss, full_loss))

               if loss < level_target:
                  break

            if loss < level_target:
               break

      print("DONE")
