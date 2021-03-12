import numpy as np
import os
import time
import matplotlib.pyplot as plt

import tensorflow as tf
import stackable_autoencoder.data
from stackable_autoencoder import models

GCP_BUCKET = "rantahar-nn"
learning_rate = 0.0008
min_learning_rate = 0.00002
lr_update_step = 100000
beta = 0.5
BATCH_SIZE = 16
IMG_SIZE = 64
ae_size = 32
n_out = 32
latent_dim = 32
steps = 3
gcl = 32
dcl = gcl

# set to True to calculate the losses from the original images
loss_from_image = False
save_every = 10000

AUTOENCODER_PATH = f'autoencoder_{ae_size}_{n_out}_{steps}'
MODEL_PATH = f'vae_{IMG_SIZE}_{dcl}_{gcl}_{n_out}'

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


# Get the first encoder and decoder levels
encoder = tf.keras.models.load_model(AUTOENCODER_PATH+"/encoder")
decoder = tf.keras.models.load_model(AUTOENCODER_PATH+"/decoder")
n_out = encoder.output_shape[-1]
img_size = IMG_SIZE
for s in range(steps):
	img_size //= 2

# A small encoder (double the latent dimension, includes variation)
small_encoder = models.make_began_encoder(2*latent_dim, gcl, n_out, size=img_size, name="e1")
small_encoder.summary()
full_encoder = models.combine_models((encoder, small_encoder))

# A small decoder
small_decoder = models.make_generator(latent_dim, gcl, n_out, size=img_size, name="d1")
small_decoder.summary()
full_decoder = models.combine_models((small_decoder, decoder))


tf_lr = tf.Variable(learning_rate)
optimizer = tf.keras.optimizers.Adam(lr=tf_lr, beta_1=beta)


if loss_from_image:
   training_encoder = full_encoder
   training_decoder = full_decoder
else:
   training_encoder = small_encoder
   training_decoder = small_decoder



def encode(x):
    mean, logvar = tf.split(training_encoder(x), num_or_size_splits=2, axis=1)
    return mean, logvar

def reparameterize(mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean



@tf.function
def train(images, batch_size):

   with tf.GradientTape() as tape:
      mean, logvar = encode(images)
      z = reparameterize(mean, logvar)
      decodings = training_decoder(z)

      encoding_loss = tf.math.reduce_sum((decodings - images)**2, axis=[1,2,3])
      #ble_loss = 0.5*tf.reduce_sum((z)**2 - (z-mean)**2*tf.exp(-logvar) - logvar)
      ble_loss = -0.5*tf.reduce_sum(logvar - tf.exp(logvar) - mean**2 , axis=1)

      encoding_loss = tf.math.reduce_mean(encoding_loss)
      ble_loss = tf.math.reduce_mean(ble_loss)
      loss = encoding_loss + 0.001*ble_loss

   gradients = tape.gradient(loss, small_encoder.trainable_variables + small_decoder.trainable_variables)
   optimizer.apply_gradients(zip(gradients, small_encoder.trainable_variables + small_decoder.trainable_variables))

   return encoding_loss, ble_loss



def save_models():
   full_encoder.save(SAVE_PATH+"/encoder")
   full_decoder.save(SAVE_PATH+"/decoder")

   noise = tf.random.normal([16, latent_dim])
   generated = full_decoder(noise)

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
      enc_loss, elb_loss  = train(element, this_batch_size)
      time_per_step = (time.time() - start_time)/s
      if s%log_step == log_step-1:
         print(' %d, %d/%d, e=%.5f, elb=%.5f, time per step=%.3f' %
            (s, j, n_batches, enc_loss, elb_loss, time_per_step))

print("DONE, saving...")
save_models()
print("saved")
