import os
import sys
import numpy as np
import subprocess
import time

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

# At the end of an encoder, flatten and map to a single latent space vector
def encoder_head(x, latent_dim):
   x = layers.Flatten()(x)
   x = layers.Dense(latent_dim*2)(x)
   x = layers.Dense(latent_dim, activation='tanh')(x)
   return x

# generate small image from a latent space vector
def decoder_head(x, size, gcl):
   n_nodes = gcl * size * size
   x = layers.Dense(gcl)(x)
   x = layers.Dense(n_nodes)(x)
   x = layers.Reshape((size, size, gcl))(x)
   return x


# Reduce the image size by a factor of 4 and encode each pixel to n_out numbers
# If the image size drops below 7, map it to a single vector
def make_encoder(shape, dcl, n_out, n_scalings = 2, latent_dim = None):
   input = tf.keras.Input(shape=shape)
   size = shape[1]
   x = input
   for i in range(n_scalings):
      if size > 7:
         x = downscale_block(x, dcl)
      else:
         output = encoder_head(x, latent_dim)
         return Model(inputs = input, outputs = output)
      size //= 2
   output = layers.Conv2D(n_out, (4,4), activation='tanh', padding='same', kernel_initializer=init)(x)
   return Model(inputs = input, outputs = output)


# Increase image size by a factor of 4 and encode each pixel to n_out numbers
# If starting from a 1D vector, first decode it into a small image
def make_decoder(input_shape, gcl, target_shape):
   n_out = target_shape[-1]
   target_size = target_shape[1] # target shape is always an image

   input = tf.keras.Input(shape=input_shape)
   if len(input_shape) == 1:
      size = target_size
      while size > 7:
         size //= 2
      x = decoder_head(input, size, gcl)
   else:
      size = input_shape[1]
      x = input

   rgb = to_rgb(x, n_out)
   while size < target_size:
      rgb, x = upscale_skip_block(rgb, x, gcl, n_out)
      size *= 2

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
   def __init__(self, shape=(64,64,3), size=32, n_out=32, n_scalings = 2, latent_dim = None, load = False, save_path=None, i=None):
      if not load:
         self.encoder = make_encoder(shape, size, n_out, n_scalings, latent_dim)
         encoding_shape = self.encoder.output_shape[1:]
         self.decoder = make_decoder(encoding_shape, size, shape)
         self.autoencoder = combine_models((self.encoder, self.decoder))
      else:
         self.load(save_path, i)

   def encoding_shape(self):
      return self.encoder.output_shape[1:]

   @tf.function
   def call(self, x):
      x = self.encoder(x)
      x = self.decoder(x)
      return x

   def save(self, path, i):
      self.encoder.save(f"{path}/encoder{i}")
      self.decoder.save(f"{path}/decoder{i}")

   def load(self, path, i):
      self.encoder = tf.keras.models.load_model(f"{path}/encoder{i}")
      self.decoder = tf.keras.models.load_model(f"{path}/decoder{i}")

   def encode(self, x):
      return self.encoder(x)

   def decode(self, x):
      return self.decoder(x)

   @tf.function
   def train_step(self, images):
      with tf.GradientTape(persistent=True) as tape:
         reproduction = self.autoencoder(images)
         loss = tf.math.reduce_mean(tf.math.square(reproduction - images))

      gradients = tape.gradient(loss, self.autoencoder.trainable_variables)
      self.optimizer.apply_gradients(zip(gradients, self.autoencoder.trainable_variables))
      return loss

   def train(self, dataset, epochs, learning_rate, beta=0.5):
      self.learning_rate = tf.Variable(learning_rate)
      self.optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate, beta_1=beta)
      n_batches = tf.data.experimental.cardinality(dataset).numpy()
      for e in range(epochs):
         start_epoch = time.time()
         start = time.time()
         for i, sample in enumerate(dataset):
            loss = self.train_step(sample)
            end = time.time()
            timing = (end - start)/float(i+1)

            sys.stdout.write(f"\repoch {e}, step {i}/{n_batches}, loss {loss}, time per step {timing}")
            sys.stdout.flush()
         sys.stdout.write("\n")
         sys.stdout.flush()





class BlockedAutoencoder():
   def __init__(self, IMG_SIZE = 64, size = 32, encoding_size = None,
                latent_dim = None, scalings_per_step = 2,
                learning_rate = 0.0001, beta = 0.9, save_path = None,
                load = False):
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
         self.save_path = f'autoencoder_{IMG_SIZE}_{size}_{encoding_size}_{scalings_per_step}'

      self.levels = []

      if not load:
         in_shape = (IMG_SIZE, IMG_SIZE, 3)
         while len(in_shape) > 1:
            autoencoder = Autoencoder(in_shape, size, encoding_size, scalings_per_step, latent_dim)
            self.levels.append(autoencoder)
            in_shape = autoencoder.encoding_shape()

      else:
         self.load()

      self.n_levels = len(self.levels)


   def save(self, bucket = None):
      for i, l in enumerate(self.levels):
         l.save(self.save_path, i)
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
      while os.path.isdir(f"{self.save_path}/decoder{i}"):
         level = Autoencoder(load = True, save_path = self.save_path, i=i)
         self.levels.append(level)
         print("loaded level", i)
         i+=1

   @tf.function
   def evaluate(self, images, level):
      x = images
      for i in range(level+1):
         x = self.levels[i].encode(x)
      for i in range(level+1):
         x = self.levels[level-i].decode(x)
      loss = tf.keras.losses.MeanSquaredError()(x, images)
      return loss

   @tf.function
   def generate(self, images, level):
      x = images
      for i in range(level+1):
         x = self.levels[i].encode(x)
      for i in range(level+1):
         x = self.levels[level-i].decode(x)
      return x


   def train(self, train_dataset, valid_dataset, epochs, bucket = None, log_step = 50,
             target_first = 0, target_increase = 0,
             save_every = 5000, learning_rate = 0.0001, lr_update_step = 10000,
             min_learning_rate = 0.00002, level = None):

      def encode_data_function(level):
         @tf.function
         def encode_data(images):
            x = images
            for i in range(level):
               x = self.levels[i].encode(x)
            return x
         return encode_data

      for l, autoencoder in enumerate(self.levels):
         x = train_dataset.map(encode_data_function(l))
         y = valid_dataset.map(encode_data_function(l))
         autoencoder.train(x, epochs, learning_rate)

         valid_image = next(iter(valid_dataset.take(1)))
         valid_loss = self.evaluate(valid_image, l)
         print(f"Full validation loss {valid_loss}")

         self.save()
