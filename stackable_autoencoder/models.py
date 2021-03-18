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

# Define some smaller utility blocks

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
   shape = tf.shape(x)
   new_shape = 2 * shape[1:3]
   x = tf.image.resize(x, new_shape, method="nearest")
   return x

# Convolution + upscale
def upscale_block(x, size):
   x = upscale(x)
   x = conv_block(x, size)
   return x

# Generate encoded output
def to_rgb(x, n_colors = 3):
   x = layers.Conv2D(n_colors, (4,4), activation='tanh', padding='same', kernel_initializer=init)(x)
   return x

# Upscale with a skip connection
def upscale_skip_block(rgb, x, size, colors = 3):
   x = upscale_block(x, size)
   rgb = upscale(rgb) + to_rgb(x, colors)
   return rgb, x

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
   x = layers.LeakyReLU(alpha=0.2)(x)
   x = layers.Dense(latent_dim, activation='tanh')(x)
   return x

# generate small image from a latent space vector
def decoder_head(x, size, gcl):
   n_nodes = gcl * size * size
   x = layers.Dense(gcl)(x)
   x = layers.LeakyReLU(alpha=0.2)(x)
   x = layers.Dense(n_nodes)(x)
   x = layers.Reshape((size, size, gcl))(x)
   return x


# Reduce the image size by a factor of 4 and encode each pixel to n_out numbers
# If the image size drops below 7, map it to a single vector
def make_encoder(shape, dcl, n_out, n_scalings = 2, latent_dim = None):
   input = tf.keras.Input(shape=shape)
   size = shape[1]
   x = input
   s = 1
   for i in range(n_scalings):
      if size is None:
         x = downscale_block(x, dcl*s)
         s *= 2
      else:
         if size > 7:
           x = downscale_block(x, dcl)
           size //= 2
         else:
           output = encoder_head(x, latent_dim)
           return Model(inputs = input, outputs = output)
   mean = layers.Conv2D(n_out, (4,4), padding='same', kernel_initializer=init)(x)
   # Could use an attention mechanism. Here we just add up a contribution from
   # each pixel
   logvar = layers.Conv2D(n_out, (4,4), padding='same')(x)
   logvar = tf.reduce_sum(logvar, axis=(1,2))
   return Model(inputs = input, outputs = [mean, logvar])


# Increase image size by a factor of 4 and encode each pixel to n_out numbers
# If starting from a 1D vector, first decode it into a small image
def make_decoder(input_shape, gcl, target_shape, n_scalings = None):
   n_out = target_shape[-1]
   target_size = target_shape[1] # target shape is always an image
   print(input_shape)
   input = tf.keras.Input(shape=input_shape)

   if target_size is None:
      x = input
      s = 2**(n_scalings-1)
      rgb = to_rgb(x, n_out)
      for i in range(n_scalings):
         rgb, x = upscale_skip_block(rgb, x, gcl*s, n_out)
         s //= 2
      model = Model(inputs = input, outputs = rgb)
      return model

   else:
      if len(input_shape) == 1:
         size = target_size
         while size > 7:
            size //= 2
         x = decoder_head(input, size, gcl)
      else:
         size = input_shape[1]
         x = input

      rgb = to_rgb(x, n_out)
      while size <= target_size//2:
         rgb, x = upscale_skip_block(rgb, x, gcl, n_out)
         size *= 2

      model = Model(inputs = input, outputs = rgb)
      return model


# A small generator for the GAN and VAE experiments
def make_generator(latent_dim, gcl, colors, size = 8, name=None):
   input = tf.keras.Input(shape=(latent_dim))
   n_nodes = gcl * size * size // 4
   x = layers.Dense(n_nodes)(input)
   x = layers.LeakyReLU(alpha=0.2)(x)
   x = layers.Reshape((size//2, size//2, gcl))(x)
   x = upscale_block(x, gcl)
   #x = conv_block(x, gcl)
   out = layers.Conv2D(colors, (4,4), padding='same', kernel_initializer=init)(x)
   return Model(inputs = input, outputs = out, name=name)

# A small encoder for the BEGAN and VAE experiments
def make_vae_encoder(latent_dim, gcl, colors, size = 8, name=None):
   input = tf.keras.Input(shape=(size, size, colors))
   x = input
   #x = conv_block(x, gcl)
   x = downscale_block(x, gcl)
   x = layers.Flatten()(x)
   x = layers.Dense(2*latent_dim)(x)
   x = layers.LeakyReLU(alpha=0.2)(x)
   mean = layers.Dense(latent_dim)(x)
   logvar = layers.Dense(latent_dim)(x)
   return Model(inputs = input, outputs = [mean, logvar], name=name)

# A small encoder for the BEGAN and VAE experiments
def make_began_encoder(latent_dim, gcl, colors, size = 8, name=None):
   input = tf.keras.Input(shape=(size, size, colors))
   x = conv_block(input, gcl)
   x = downscale_block(x, gcl)
   x = layers.Flatten()(x)
   output = layers.Dense(latent_dim)(x)
   return Model(inputs = input, outputs = output, name=name)

# A small discriminator for the WGAN
def make_wgan_discriminator(gcl, colors, size = 8, name=None):
   input = tf.keras.Input(shape=(size, size, colors))
   x = downscale_block(input, gcl)
   x = layers.Flatten()(x)
   output = layers.Dense(1)(x)
   return Model(inputs = input, outputs = output, name=name)


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

def combine_with_encoder(encoder, extention):
   input_shape = encoder.input_shape[1:]
   input = tf.keras.Input(shape=input_shape)
   mean, _ = encoder(input)
   output = extention(mean)
   model = Model(inputs = input, outputs = output)
   return model


class Autoencoder():
   def __init__(self, shape=(None,None,3), size=32, n_out=32, n_scalings = 2, latent_dim = None, load = False, save_path=None):

      if save_path is not None:
         self.save_path = save_path
      else:
         self.save_path = f'svae_{size}_{n_out}_{n_scalings}'
         print(self.save_path)
      if not load:
         self.encoder = make_encoder(shape, size, n_out, n_scalings, latent_dim)
         enc_size = self.encoder.output_shape[0][1:]
         self.decoder = make_decoder(enc_size, size, shape, n_scalings)
         self.encoder.summary(line_length=150)
         self.decoder.summary(line_length=150)
      else:
         self.load()
      self.trainable_variables = self.encoder.trainable_variables + self.decoder.trainable_variables

   def encoding_shape(self):
      enc_size = self.encoder.output_shape[0]
      return enc_size

   @tf.function
   def encode(self, x):
      mean, logvar = self.encoder(x)
      return mean, logvar

   @tf.function
   def reparameterize(self, mean, logvar):
      eps = tf.random.normal(shape=logvar.shape) * tf.exp(0.5*logvar)
      s = mean.shape[1]
      eps = tf.reshape(eps, (logvar.shape[0],1,1,logvar.shape[1]))
      eps = tf.tile(eps, [1,s,s,1])
      return eps + mean

   @tf.function
   def decode(self, x):
      return self.decoder(x)

   @tf.function
   def call(self, x):
      mean, logvar = self.encode(x)
      x = self.decoder(mean)
      return x

   def save(self):
      self.encoder.save(f"{self.save_path}/encoder")
      self.decoder.save(f"{self.save_path}/decoder")

   def load(self):
      self.encoder = tf.keras.models.load_model(f"{self.save_path}/encoder")
      self.decoder = tf.keras.models.load_model(f"{self.save_path}/decoder")

   @tf.function
   def train_step(self, images):
      with tf.GradientTape() as tape:
         mean, logvar = self.encode(images)
         z = self.reparameterize(mean, logvar)
         decodings = self.decode(z)

         encoding_loss = tf.math.reduce_mean(tf.math.square(decodings - images))
         elb_loss = 0.5*tf.math.reduce_mean(mean**2) + 0.5*tf.math.reduce_mean(tf.exp(logvar) - logvar)

         loss = encoding_loss + 0.01*elb_loss

      gradients = tape.gradient(loss, self.trainable_variables)
      self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
      return encoding_loss, elb_loss

   def train(self, dataset, valid_dataset = None, epochs=1, batches=None, log_step = 1, learning_rate = 0.0001, beta=0.5, save_every = None):
      self.learning_rate = tf.Variable(learning_rate)
      self.optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate, beta_1=beta)
      if batches is None:
         batches = tf.data.experimental.cardinality(dataset).numpy()
      for e in range(epochs):
         start_epoch = time.time()
         start = time.time()
         for i, sample in enumerate(dataset):
            loss, elb = self.train_step(sample)
            end = time.time()

            if i % log_step == log_step - 1:
               timing = (end - start)/float(i+1)
               print(f"epoch {e}, step {i}/{batches}, loss {loss}, elb {elb}, time per step {timing}")
            if save_every is not None and i % save_every == save_every - 1:
               self.save()

         if valid_dataset is not None:
            validation_loss = 0
            for i, sample in enumerate(valid_dataset.take(100)):
               validation_loss += self.evaluate_batch(sample)
            print(f"epoch {e}, validation loss {validation_loss/100}")

   @tf.function
   def evaluate_batch(self, images):
      mean, logvar = self.encode(images)
      decodings = self.decode(mean)
      loss = tf.keras.losses.MeanSquaredError()(decodings, images)
      return loss

   def evaluate(self, dataset):
      n_batches = tf.data.experimental.cardinality(dataset).numpy()
      loss = 0
      for images in dataset:
         loss += self.evaluate_batch(images)
      return loss / n_batches




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


   def save(self):
      for i, l in enumerate(self.levels):
         l.save(f"{self.save_path}/level{i}")

   def load(self):
      self.levels = []
      i = 0
      while os.path.isdir(f"{self.save_path}/level{i}"):
         level = Autoencoder(load = True, save_path = f"{self.save_path}/level{i}")
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


   def train(self, train_dataset, valid_dataset, epochs, log_step = 50,
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
         autoencoder.train(x, epochs, learning_rate=learning_rate)

         valid_image = next(iter(valid_dataset.take(1)))
         valid_loss = self.evaluate(valid_image, l)
         print(f"Full validation loss {valid_loss}")

         self.save()
