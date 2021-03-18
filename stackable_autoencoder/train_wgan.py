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
beta = 0.5
BATCH_SIZE = 16
IMG_SIZE = 64
critic_updates = 5
gp_weight = 10
ae_size = 32
n_out = 32
steps = 3
latent_dim = 32
gcl = 32
dcl = gcl*2

# set to True to calculate the losses from the original images
loss_from_image = False
save_every = 10000

AUTOENCODER_PATH = f'autoencoder_{ae_size}_{n_out}_{steps}'
MODEL_PATH = f'wgan_{IMG_SIZE}_{dcl}_{gcl}_{n_out}'

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
small_discriminator = models.make_wgan_discriminator(gcl, n_out, size=img_size, name="d1")
small_discriminator.summary()
discriminator = models.combine_models((encoder, small_discriminator))

# generator: just a small decoder
small_generator = models.make_generator(latent_dim, gcl, n_out, size=img_size)
small_generator.summary()
generator = models.combine_models((small_generator, decoder))

tf_lr = tf.Variable(learning_rate)
disc_optimizer = tf.keras.optimizers.Adam(lr=tf_lr, beta_1=beta)
gen_optimizer = tf.keras.optimizers.Adam(lr=tf_lr, beta_1=beta)


if loss_from_image:
   training_disciminator = discriminator
   training_generator = generator
else:
   training_disciminator = small_discriminator
   training_generator = small_generator


def gradient_penalty(fake, real, batch_size):
   alpha = tf.random.uniform(shape=tf.shape(fake), minval=0., maxval=1.)
   diff = fake - real
   interpolation = real + (alpha * diff)
   with tf.GradientTape() as tape:
      tape.watch(interpolation)
      quality = training_disciminator(interpolation)
   gradients = tape.gradient(quality, interpolation)
   d = tf.sqrt(tf.math.reduce_sum(gradients**2,  [1, 2, 3])+1e-12) - 1.0
   penalty = tf.math.reduce_mean(d**2)
   return penalty


@tf.function
def train_discriminator(images, batch_size):
   noise = tf.random.normal([batch_size, latent_dim])

   with tf.GradientTape() as tape:
      fake_images = training_generator(noise, training=False)
      real_quality = training_disciminator(images)
      fake_quality = training_disciminator(fake_images)
      g_loss = tf.math.reduce_mean(fake_quality)
      d_loss = tf.math.reduce_mean(real_quality)
      loss = d_loss - g_loss

      gp = gradient_penalty(fake_images, images, batch_size)
      loss = d_loss + gp_weight*gp

   discriminator_gradients = tape.gradient(loss, small_discriminator.trainable_variables)
   disc_optimizer.apply_gradients(zip(discriminator_gradients, small_discriminator.trainable_variables))

   return loss, g_loss, gp


@tf.function
def train_generator():
   noise = tf.random.normal([BATCH_SIZE, latent_dim])

   with tf.GradientTape() as tape:
	   fake_images = training_generator(noise)
	   fake_quality = training_disciminator(fake_images, training=False)
	   loss = tf.math.reduce_mean(fake_quality)

   gradients = tape.gradient(loss, small_generator.trainable_variables)
   gen_optimizer.apply_gradients(zip(gradients, small_generator.trainable_variables))


def save_models():
   discriminator.save(SAVE_PATH+"/discriminator")
   generator.save(SAVE_PATH+"/generator")

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
      real_loss, fake_loss, gradient_loss = train_discriminator(element, this_batch_size)
      if s%critic_updates == 0:
         train_generator()
      time_per_step = (time.time() - start_time)/s
      if s%log_step == log_step-1:
         print(' %d, %d/%d, r1=%.5f, g=%.5f, gp=%.3f, time per step=%.3f' %
            (s, j, n_batches, real_loss, fake_loss, gradient_loss, time_per_step))

print("DONE, saving...")
save_models()
print("saved")
