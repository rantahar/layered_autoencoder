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
import stackable_autoencoder.data
from stackable_autoencoder import models

GCP_BUCKET = "rantahar-nn"
learning_rate = 0.0001
min_learning_rate = 0.00002
lr_update_step = 10000
gamma = 0.5
lambda_Kt = 0.001
id_weight = 0.1
beta = 0.5
BATCH_SIZE = 16
IMG_SIZE = 64
encoded_size = 8
dcl = 64
gcl = 64
latent_dim = 64

AUTOENCODER_PATH = f'autoencoder_{IMG_SIZE}_64_3'
MODEL_PATH = f'began_{IMG_SIZE}_{dcl}_{gcl}_{latent_dim}'

# Specific training parameters
samples = 500000
SAVE_PATH = MODEL_PATH
DATA_PATH = 'celeba'
save_every = 5000
log_step = 1

remote = False

if remote:
	print("Downloading data")
	cmd = [
    	'gsutil', '-m', 'cp', '-r',
    	os.path.join('gs://', GCP_BUCKET, DATA_PATH+'.zip'),
    	'./'
	]
	print(subprocess.list2cmdline(cmd))
	subprocess.call(cmd)
	cmd = [
    	'unzip', DATA_PATH+'.zip'
	]
	print(subprocess.list2cmdline(cmd))
	subprocess.call(cmd)
else:
   save_every = 5000
   DATA_PATH = '../data/' + DATA_PATH


dataset, _ = stackable_autoencoder.data.get_celeba(IMG_SIZE, BATCH_SIZE)
n_batches = tf.data.experimental.cardinality(dataset)
epochs = samples//n_batches + 1



# Get the first encoder and decoder levels
encoder = tf.keras.models.load_model(AUTOENCODER_PATH+"/encoder")
decoder = tf.keras.models.load_model(AUTOENCODER_PATH+"/decoder")
n_out = encoder.output_shape[-1]
size = encoder.output_shape[1]

# dicriminator: combine small encoder and decoder
small_encoder = models.make_began_encoder(latent_dim, gcl, n_out, size=encoded_size)
small_encoder.summary()
small_decoder = models.make_generator(latent_dim, gcl, n_out, size=encoded_size)
small_decoder.summary()

small_discriminator = models.combine_models((small_encoder, small_decoder))

# full dicscriminator
discriminator = models.combine_models((encoder, small_discriminator, decoder))

# generator: just a small decoder
small_generator = models.make_generator(latent_dim, gcl, n_out, size=encoded_size)
small_generator.summary()

# full generator
generator = models.combine_models((small_generator, decoder))


tf_lr = tf.Variable(learning_rate)
disc_optimizer = tf.keras.optimizers.Adam(lr=tf_lr, beta_1=beta)
gen_optimizer = tf.keras.optimizers.Adam(lr=tf_lr, beta_1=beta)



@tf.function
def train_discriminator(images, batch_size, Kt):
   z = tf.random.uniform([batch_size, latent_dim], minval=-1)
   image_encodings = encoder(images)

   with tf.GradientTape() as tape:
      real_quality = small_discriminator(image_encodings)
      real_loss = tf.math.reduce_mean(tf.math.abs(real_quality - image_encodings))

      fake_encodings = small_generator(z)
      fake_qualities = small_discriminator(fake_encodings)
      z_d = small_encoder(fake_encodings)

      fake_loss = tf.math.reduce_mean(tf.math.abs(fake_encodings - fake_qualities))
      id_loss = tf.math.reduce_mean(tf.math.abs(z_d - z))
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
      fake_encodings = small_generator(z)
      fake_qualities = small_discriminator(fake_encodings)
      loss = tf.math.reduce_mean(tf.math.abs(fake_encodings - fake_qualities))

   gradients = tape.gradient(loss, small_generator.trainable_variables)
   gen_optimizer.apply_gradients(zip(gradients, small_generator.trainable_variables))


def save_models():
   discriminator.save(SAVE_PATH+"/discriminator")
   generator.save(SAVE_PATH+"/generator")
   if remote:
      print("Uploading model")
      subprocess.call([
    		'gsutil', 'cp', '-r',
			os.path.join(SAVE_PATH),
    		os.path.join('gs://', GCP_BUCKET)
      ])

# train the discriminator and decoder
Kt = 0
s = 0
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
      if s%log_step == log_step-1:
         print(' %d, %d/%d, r1=%.3f, g=%.3f, e=%.3f, Kt=%.3f, convergence=%.3f' %
            (s, j, n_batches, real_loss, fake_loss, id_loss, Kt, convergence))

print("DONE, saving...")
save_models()
print("saved")
