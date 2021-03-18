import numpy as np
import os
import time
import matplotlib.pyplot as plt

import tensorflow as tf
import stackable_autoencoder.data
from stackable_autoencoder import models

GCP_BUCKET = "rantahar-nn"
learning_rate = 0.00008
min_learning_rate = 0.00002
lr_update_step = 100000
gamma = 0.1
lambda_Kt = 0.001
constraint_weight = 0.001
beta = 0.5
BATCH_SIZE = 16
IMG_SIZE = 64
ae_size = 32
n_out = 64
latent_dim = 128
steps = 3
gcl = 64
dcl = 128

loss_from_image = False
continue_training = False
save_every = 1000

AUTOENCODER_PATH = f'gsvae_{ae_size}_{n_out}_{steps}'
MODEL_PATH = f'began_{IMG_SIZE}_{dcl}_{gcl}_{n_out}'

if loss_from_image:
   DATA_PATH = '../data/celeba'
else:
   DATA_PATH = f'celeba_{ae_size}_{n_out}_{steps}.npy'

samples = 300000
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


# Get the encoder and decoder
autoencoder = models.Autoencoder(load = True, save_path = AUTOENCODER_PATH)
n_out = autoencoder.encoding_shape()[-1]
img_size = IMG_SIZE
for s in range(steps):
   img_size //= 2


if not continue_training:
   # discriminator: combine small encoder and decoder
   small_encoder = models.make_began_encoder(latent_dim, dcl, n_out, size=img_size, name="e1")
   small_encoder.summary()

   # A small decoder
   small_decoder = models.make_generator(latent_dim, gcl, n_out, size=img_size, name="d2")
   small_decoder.summary()

   # The discriminator
   small_discriminator = models.combine_models((small_encoder, small_decoder))

   # generator: just a small decoder
   small_generator = models.make_generator(latent_dim, gcl, n_out, size=img_size, name="g1")
   small_generator.summary()

else:
   small_encoder = tf.keras.models.load_model(SAVE_PATH+"/small_encoder")
   small_decoder = tf.keras.models.load_model(SAVE_PATH+"/small_decoder")
   small_generator = tf.keras.models.load_model(SAVE_PATH+"/small_generator")


full_encoder = models.combine_with_encoder(autoencoder.encoder, small_encoder)
full_decoder = models.combine_models((small_decoder, autoencoder.decoder))
discriminator = models.combine_models((full_encoder, full_decoder))
generator = models.combine_models((small_generator, autoencoder.decoder))


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
   z = tf.random.uniform([batch_size, latent_dim])

   with tf.GradientTape() as tape:
      real_quality = training_disciminator(images)
      real_loss = tf.math.reduce_mean(tf.math.reduce_sum((real_quality - images)**2, axis=3))

      fake_encodings = training_generator(z)
      fake_qualities = training_disciminator(fake_encodings)
      z_d = training_encoder(fake_encodings)

      fake_loss = tf.math.reduce_mean(tf.math.reduce_sum((fake_encodings - fake_qualities)**2, axis=3))
      contraint_loss = tf.math.reduce_mean((z_d - z)**2)
      loss = real_loss - Kt * fake_loss + constraint_weight * contraint_loss

   gradients = tape.gradient(loss, small_discriminator.trainable_variables)
   disc_optimizer.apply_gradients(zip(gradients, small_discriminator.trainable_variables))

   Kt = Kt + lambda_Kt * (gamma * real_loss - fake_loss)
   if Kt < 0.001:
      Kt = 0.001
   if Kt > 1.0:
      Kt = 1.0

   convergence = real_loss + tf.abs(gamma * real_loss - fake_loss)

   return real_loss, fake_loss, contraint_loss, Kt, convergence


@tf.function
def train_generator():
   z = tf.random.uniform([BATCH_SIZE, latent_dim], minval=-1)

   with tf.GradientTape() as tape:
      fake_encodings = training_generator(z)
      fake_qualities = training_disciminator(fake_encodings)
      loss = tf.math.reduce_mean((fake_encodings - fake_qualities)**2)

   gradients = tape.gradient(loss, small_generator.trainable_variables)
   gen_optimizer.apply_gradients(zip(gradients, small_generator.trainable_variables))


def save_models():
   discriminator.save(SAVE_PATH+"/discriminator")
   generator.save(SAVE_PATH+"/generator")
   small_encoder.save(SAVE_PATH+"/small_encoder")
   small_decoder.save(SAVE_PATH+"/small_decoder")
   small_generator.save(SAVE_PATH+"/small_generator")

   noise = tf.random.uniform([16, latent_dim], minval=-1)
   generated = generator(noise)

   global s
   fig, arr = plt.subplots(4, 4)
   for i in range(16):
      im = (generated[i] + 1) / 2
      arr[i%4, i//4].imshow(im)
   for ax in fig.axes:
       ax.axis("off")
   fig.tight_layout()
   fig.savefig(f'sample_{s}.png')
   plt.close(fig)

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

      this_batch_size = element.shape[0]
      real_loss, fake_loss, c_loss, Kt, convergence = train_discriminator(element, this_batch_size, Kt)
      train_generator()
      time_per_step = (time.time() - start_time)/s
      if s%log_step == log_step-1:
         print(' %d, %d/%d, r1=%.5f, g=%.5f, e=%.3f, Kt=%.3f, convergence=%.3f, time per step=%.3f' %
            (s, j, n_batches, real_loss, fake_loss, c_loss, Kt, convergence, time_per_step))

print("DONE, saving...")
save_models()
print("saved")
