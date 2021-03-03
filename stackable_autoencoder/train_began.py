import numpy as np
import os
import subprocess
import time

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras import backend
from tensorflow.keras.initializers import RandomNormal

from tensorflow.keras.preprocessing import image_dataset_from_directory
import stackable_autoencoder.data
from stackable_autoencoder import models

GCP_BUCKET = "rantahar-nn"
learning_rate = 0.00008
min_learning_rate = 0.00002
lr_update_step = 100000
gamma = 0.5
lambda_Kt = 0.001
id_weight = 0.1
beta = 0.5
BATCH_SIZE = 16
IMG_SIZE = 64
ae_size = 64
n_out = 64
latent_dim = 128
steps = 3
dcl = 64
gcl = 64

# set to True to calculate the losses from the original images
loss_from_image = True
save_every = 10

AUTOENCODER_PATH = f'autoencoder_{ae_size}_{n_out}_{steps}'
MODEL_PATH = f'began_{IMG_SIZE}_{dcl}_{gcl}_{n_out}'

if loss_from_image:
   DATA_PATH = '../data/celeba'
else:
   DATA_PATH = f'celeba_{ae_size}_{n_out}_{steps}.npy'

samples = 500000
SAVE_PATH = MODEL_PATH
log_step = 1


if loss_from_image:
   dataset = stackable_autoencoder.data.dataset_from_folder(DATA_PATH, IMG_SIZE, BATCH_SIZE)
else:
   data = np.load(DATA_PATH)
   dataset = tf.data.Dataset.from_tensor_slices(data).shuffle(128).batch(BATCH_SIZE)
   del data

n_batches = tf.data.experimental.cardinality(dataset)
epochs = samples//n_batches + 1


# Get the first encoder and decoder levels
encoder = tf.keras.models.load_model(AUTOENCODER_PATH+"/encoder")
decoder = tf.keras.models.load_model(AUTOENCODER_PATH+"/decoder")
n_out = encoder.output_shape[-1]
img_size = IMG_SIZE
for s in range(steps):
	img_size //= 2

# dicriminator: combine small encoder and decoder
small_encoder = models.make_began_encoder(latent_dim, gcl, n_out, size=img_size, name="e1")
small_encoder.summary()
full_encoder = models.combine_models((encoder, small_encoder))
small_decoder = models.make_generator(latent_dim, gcl, n_out, size=img_size, name="d1")
small_decoder.summary()
full_decoder = models.combine_models((small_decoder, decoder))

small_discriminator = models.combine_models((small_encoder, small_decoder))

# full dicscriminator
discriminator = models.combine_models((encoder, small_discriminator, decoder))

# generator: just a small decoder
small_generator = models.make_generator(latent_dim, gcl, n_out, size=img_size)
small_generator.summary()

# full generator
generator = models.combine_models((small_generator, decoder))

tf_lr = tf.Variable(learning_rate)
disc_optimizer = tf.keras.optimizers.Adam(lr=tf_lr, beta_1=beta)
gen_optimizer = tf.keras.optimizers.Adam(lr=tf_lr, beta_1=beta)


if loss_from_image:
   training_disciminator = discriminator
   training_generator = generator
   training_encoder = full_encoder
else:
   training_disciminator = small_discriminator
   training_generator = small_generator
   training_encoder = small_encoder


@tf.function
def train_discriminator(images, batch_size, Kt):
   z = tf.random.uniform([batch_size, latent_dim], minval=-1)

   with tf.GradientTape() as tape:
      real_quality = training_disciminator(images)
      real_loss = tf.math.reduce_mean(tf.math.reduce_sum(tf.math.square(real_quality - images), axis=3))

      fake_encodings = training_generator(z)
      fake_qualities = training_disciminator(fake_encodings)
      z_d = training_encoder(fake_encodings)

      fake_loss = tf.math.reduce_mean(tf.math.reduce_sum(tf.math.square(fake_encodings - fake_qualities), axis=3))
      id_loss = tf.math.reduce_mean(tf.math.square(z_d - z))
      loss = real_loss - Kt * fake_loss + id_weight * id_loss

   gradients = tape.gradient(loss, small_discriminator.trainable_variables)
   disc_optimizer.apply_gradients(zip(gradients, small_discriminator.trainable_variables))

   Kt = Kt + lambda_Kt * (gamma * real_loss - fake_loss)
   if Kt < 0.001:
      Kt = 0.001
   if Kt > 1.0:
      Kt = 1.0

   convergence = real_loss + tf.abs(gamma * real_loss - fake_loss)

   return real_loss, fake_loss, id_loss, Kt, convergence


@tf.function
def train_generator():
   z = tf.random.uniform([BATCH_SIZE, latent_dim], minval=-1)

   with tf.GradientTape() as tape:
      fake_encodings = training_generator(z)
      fake_qualities = training_disciminator(fake_encodings)
      loss = tf.math.reduce_mean(tf.math.square(fake_encodings - fake_qualities))

   gradients = tape.gradient(loss, small_generator.trainable_variables)
   gen_optimizer.apply_gradients(zip(gradients, small_generator.trainable_variables))


def save_models():
   discriminator.save(SAVE_PATH+"/discriminator")
   generator.save(SAVE_PATH+"/generator")

# train the discriminator and decoder
Kt = 0
s = 0
start_time = time.time()
# manually enumerate epochs
for i in range(epochs):
   for element in dataset:
      s += 1
      j = s%n_batches

      if s%save_every == save_every-1:
         save_models()
         print("saved")

      if s%lr_update_step == lr_update_step-1:
         if learning_rate > min_learning_rate:
            learning_rate = learning_rate/2
            tf_lr.assign(learning_rate)

      this_batch_size = element[0].shape[0]
      real_loss, fake_loss, id_loss, Kt, convergence = train_discriminator(element, this_batch_size, Kt)
      train_generator()
      time_per_step = (time.time() - start_time)/s
      if s%log_step == log_step-1:
         print(' %d, %d/%d, r1=%.5f, g=%.5f, e=%.3f, Kt=%.3f, convergence=%.3f, time per step=%.3f' %
            (s, j, n_batches, real_loss, fake_loss, id_loss, Kt, convergence, time_per_step))

print("DONE, saving...")
save_models()
print("saved")
