import numpy as np
import os
import subprocess

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras import backend
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.initializers import RandomNormal

from tensorflow.keras.preprocessing import image_dataset_from_directory

MODEL_PATH = 'began_disc_16_16'
GCP_BUCKET = "rantahar-nn"
learning_rate = 0.0001
min_learning_rate = 0.00002
lr_update_step = 100000
gamma = 0.5
lambda_Kt = 0.001
reproduction_weight = 0.1
beta = 0.5
BATCH_SIZE = 16
IMG_SIZE = 64
dcl = 16
gcl = 16
latent_dim = 64
disc_encoding_size = 16

# Specific training parameters
epochs = 50
SAVE_PATH = MODEL_PATH # for local
DATA_PATH = 'celeba'
save_every = 5000
log_step = 50

remote = True

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
   save_every = 200
   log_step = 1
   DATA_PATH = '../data/' + DATA_PATH

dataset = image_dataset_from_directory(DATA_PATH,
                                       shuffle=True,
                                       batch_size=BATCH_SIZE,
                                       image_size=(IMG_SIZE,IMG_SIZE))


def normalize(image, label):
   image = tf.cast(image, tf.float32)
   image = (image / 127.5) - 1
   return image, label

dataset = dataset.map(normalize)



init = RandomNormal(stddev=0.02)

def upscale_block(x, size, upscale = True):
   x = layers.Conv2D(gcl, (4,4), padding='same', kernel_initializer=init)(x)
   x = layers.BatchNormalization()(x)
   x = layers.LeakyReLU(alpha=0.2)(x)
   #x = layers.Conv2D(gcl, (4,4), padding='same', kernel_initializer=init)(x)
   #x = layers.BatchNormalization()(x)
   #x = layers.LeakyReLU(alpha=0.2)(x)
   if upscale:
      img_size = x.shape[1]
      x = tf.image.resize(x, (2*img_size, 2*img_size), method="nearest")
   return x

def downscale_block(x, size, downscale = True):
   x = layers.Conv2D(gcl, (4,4), padding='same', kernel_initializer=init)(x)
   x = layers.BatchNormalization()(x)
   x = layers.LeakyReLU(alpha=0.2)(x)
   #x = layers.Conv2D(gcl, (4,4), padding='same', kernel_initializer=init)(x)
   #x = layers.BatchNormalization()(x)
   #x = layers.LeakyReLU(alpha=0.2)(x)
   if downscale:
      x = layers.Conv2D(gcl, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(x)
      x = layers.BatchNormalization()(x)
      x = layers.LeakyReLU(alpha=0.2)(x)
   return x


n_steps = 0
size = 4
while size < IMG_SIZE:
   size*=2
   n_steps += 1

def make_generator():
   input = tf.keras.Input(shape=(latent_dim))
   n_nodes = gcl * 4 * 4
   x = layers.Dense(n_nodes)(input)
   x = layers.Reshape((4, 4, gcl))(x)
   for i in range(n_steps):
      x = upscale_block(x, gcl)
   x = upscale_block(x, gcl, upscale = False)
   output = layers.Conv2DTranspose(3, (4,4), activation='tanh', padding='same', kernel_initializer=init)(x)
   model = Model(inputs = input, outputs = output)
   return model

def make_encoder():
   input = tf.keras.Input(shape=(IMG_SIZE,IMG_SIZE,3))
   x = input
   s=1
   for i in range(n_steps):
      x = downscale_block(x, dcl*s)
      s*=2
   x = downscale_block(x,  dcl*s, downscale = False)
   x = layers.Flatten()(x)
   output = layers.Dense(latent_dim, activation='tanh')(x)
   model = Model(inputs = input, outputs = output)
   return model



def make_discriminator():
   input = tf.keras.Input(shape=(IMG_SIZE,IMG_SIZE,3))
   # encoding
   x = input
   s = 1
   for i in range(4):
      x = downscale_block(x, dcl*s)
      s*=2
   x = downscale_block(x,  dcl*s, downscale = False)
   x = layers.Conv2D(disc_encoding_size, (4,4), padding='same', kernel_initializer=init, activation='tanh')(x)

   # IMG_SIZE/16 ** 2 * disc_encoding_size = 4**2 * 8

   # decoding
   for i in range(4):
      x = downscale_block(x, gcl)
      s*=2
   x = upscale_block(x, gcl, upscale = False)
   output = layers.Conv2DTranspose(3, (4,4), activation='tanh', padding='same', kernel_initializer=init)(x)
   model = Model(inputs = input, outputs = output)
   return model

discriminator = make_discriminator()
discriminator.summary()
generator = make_generator()
generator.summary()
encoder = make_encoder()
encoder.summary()


tf_lr = tf.Variable(learning_rate)
discriminator_optimizer = tf.keras.optimizers.Adam(lr=tf_lr, beta_1=beta)
generator_optimizer = tf.keras.optimizers.Adam(lr=tf_lr, beta_1=beta)
encoder_optimizer = tf.keras.optimizers.Adam(lr=tf_lr, beta_1=beta)




@tf.function
def train(images, batch_size, Kt):
   noise = tf.random.normal([batch_size, latent_dim])

   with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
      fake_images = generator(noise)
      real_quality = discriminator(images)
      fake_quality = discriminator(fake_images)

      encoding = encoder(images)
      reproduction = generator(encoding)

      real_loss = tf.math.reduce_mean(tf.math.abs(real_quality - images))
      fake_loss = tf.math.reduce_mean(tf.math.abs(fake_quality - fake_images))
      repr_loss = tf.math.reduce_mean(tf.math.abs(reproduction - images))

      d_loss  = real_loss - Kt * fake_loss
      g_loss  = fake_loss + reproduction_weight * repr_loss

   generator_gradients = gen_tape.gradient(g_loss, generator.trainable_variables)
   generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
   discriminator_gradients = disc_tape.gradient(d_loss, discriminator.trainable_variables)
   discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

   Kt = Kt + lambda_Kt * (gamma * real_loss - fake_loss)
   if Kt < 0.0:
      Kt = 0.0
   if Kt > 1.0:
      Kt = 1.0

   convergence = real_loss + tf.abs(gamma * real_loss - fake_loss)

   return fake_loss, repr_loss, real_loss, Kt, convergence


def save_models():
   discriminator.save(SAVE_PATH+"/discriminator")
   generator.save(SAVE_PATH+"/generator")
   generator.save(SAVE_PATH+"/encoder")
   if remote:
      print("Uploading model")
      subprocess.call([
    		'gsutil', 'cp', '-r',
			os.path.join(SAVE_PATH),
    		os.path.join('gs://', GCP_BUCKET)
      ])

# train the discriminator and decoder
n_batches = tf.data.experimental.cardinality(dataset)
Kt = 0
s=0
# manually enumerate epochs
for i in range(epochs):
   for j, element in enumerate(dataset):
      s = s+1
      if j%save_every == 0:
         save_models()
         print("saved")
      if s%lr_update_step == lr_update_step-1:
         if learning_rate > min_learning_rate:
            learning_rate = learning_rate/2
            tf_lr.assign(learning_rate)
      this_batch_size = element[0].shape[0]
      fake_loss, repr_loss, real_loss, Kt, convergence = train(element[0], this_batch_size, Kt)
      if s%log_step == log_step-1:
         print(' %d, %d/%d, f=%.3f, repr=%.3f, r=%.3f, Kt=%.3f, convergence=%.3f' %
            (s, j+1, n_batches, fake_loss, repr_loss, real_loss, Kt, convergence))


save_models()
