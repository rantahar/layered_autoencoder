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

GCP_BUCKET = "rantahar-nn"
learning_rate = 0.0008
min_learning_rate = 0.00002
lr_update_step = 10000
gamma = 0.5
lambda_Kt = 0.001
id_weight = 0.1
beta = 0.5
BATCH_SIZE = 16
IMG_SIZE = 64
dcl = 16
gcl = 16
latent_dim = 64

MODEL_PATH = f'edbegan_{IMG_SIZE}_{dcl}_{gcl}'

# Specific training parameters
samples = 50000
SAVE_PATH = MODEL_PATH
DATA_PATH = 'celeba'
save_every = 5000
log_step = 50

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
   save_every = 100
   log_step = 1
   DATA_PATH = '../data/' + DATA_PATH


def normalize(image, label):
   image = tf.cast(image, tf.float32)
   image = (image / 127.5) - 1
   return image, label

def flip(image, label):
   tf.image.random_flip_left_right(image)
   return image, label

dataset = image_dataset_from_directory(DATA_PATH,
                                       batch_size=1,
                                       image_size=(IMG_SIZE,IMG_SIZE))
n_images = tf.data.experimental.cardinality(dataset)

dataset = dataset.unbatch().map(normalize).cache().map(flip).shuffle(100).batch(BATCH_SIZE)
n_batches = n_images//BATCH_SIZE

init = RandomNormal(stddev=0.02)

def conv_block(x, size):
   x = layers.Conv2D(size, (4,4), padding='same', kernel_initializer=init)(x)
   x = layers.BatchNormalization()(x)
   x = layers.LeakyReLU(alpha=0.2)(x)
   #x = layers.Conv2D(size, (4,4), padding='same', kernel_initializer=init)(x)
   #x = layers.BatchNormalization()(x)
   #x = layers.LeakyReLU(alpha=0.2)(x)
   return x

def upscale(x):
   img_size = x.shape[1]
   x = tf.image.resize(x, (2*img_size, 2*img_size), method="nearest")
   return x

def upscale_block(x, size):
   x = conv_block(x, size)
   x = upscale(x)
   return x

def downscale_block(x, size):
   x = conv_block(x, size)
   x = layers.Conv2D(size, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(x)
   x = layers.LeakyReLU(alpha=0.2)(x)
   x = layers.BatchNormalization()(x)
   return x

def to_rpg(x, n_colors = 3):
   x = layers.Conv2D(n_colors, (4,4), activation='tanh', padding='same', kernel_initializer=init)(x)
   return x


def make_generator(n_in, gcl, n_out):
   input = tf.keras.Input(shape=(n_in))
   n_nodes = gcl * 4 * 4
   x = layers.Dense(n_nodes)(input)
   x = layers.Reshape((4, 4, gcl))(x)
   x = upscale_block(x, gcl)
   images = [to_rpg(x)]
   size = 8
   while size < IMG_SIZE:
      x = upscale_block(x, gcl)
      images = [upscale(i) for i in images]
      images.append(to_rpg(x))
      size *= 2
   model = Model(inputs = input, outputs = images)
   return model


g1 = make_generator(latent_dim, gcl, 3)
g1.summary()

def make_encoder(input_shape, gcl, latent_dim):
   features = input_shape[-1]
   input_size = input_shape[1]
   input = tf.keras.Input(shape=input_shape)
   x = input
   size = 64
   s = 1
   while size > 4:
      x = downscale_block(x, gcl*s)
      s*=2
      size /= 2
   x = layers.Flatten()(x)
   x = layers.Dense(latent_dim*2)(x)
   output = layers.Dense(latent_dim)(x)
   model = Model(inputs = input, outputs = output)
   print(g1.output_shape)
   return model
e1 = make_encoder(g1.output_shape[-1][1:], gcl, latent_dim)
e1.summary()

def make_discriminator(input_shape, dcl, latent_dim):
   features = input_shape[-1]
   input_size = input_shape[1]
   input = tf.keras.Input(shape=input_shape)
   # encoding
   x = input
   size = 64
   s = 1
   while size > 4:
      x = downscale_block(x, gcl*s)
      s*=2
      size /= 2
   x = conv_block(x, dcl*s)
   x = conv_block(x, dcl*s)
   x = conv_block(x, dcl*s)
   x = conv_block(x, dcl*s)
   x = upscale_block(x, dcl)
   size = 8
   while size < IMG_SIZE:
      x = upscale_block(x, dcl)
      size*=2
   rpg = to_rpg(x)
   output = layers.Conv2DTranspose(features, (4,4), activation='tanh', padding='same', kernel_initializer=init)(x)
   model = Model(inputs = input, outputs = output)
   return model


d1 = make_discriminator(g1.output_shape[-1][1:], dcl, latent_dim)
d1.summary()


tf_lr = tf.Variable(learning_rate)
d1_optimizer = tf.keras.optimizers.Adam(lr=tf_lr, beta_1=beta)
g1_optimizer = tf.keras.optimizers.Adam(lr=tf_lr, beta_1=beta)
e1_optimizer = tf.keras.optimizers.Adam(lr=tf_lr, beta_1=beta)



@tf.function
def train(images, batch_size, Kt):
   noise = tf.random.uniform([batch_size, latent_dim], minval=-1)

   with tf.GradientTape(persistent=True) as tape:
      fake_images = g1(noise)

      real_image_quality = d1(images)
      fake_qualities  = [d1(i) for i in fake_images]

      real_loss = tf.math.reduce_mean(tf.math.abs(real_image_quality - images))
      fake_loss = 0
      for i, q in zip(fake_images, fake_qualities):
         fake_loss = 0.5*fake_loss + 0.5*tf.math.reduce_mean(tf.math.abs(i - q))

      # Reproduce original noise to reduce mode collapse
      noise_repr = e1(fake_images[-1])
      image_repr = g1(e1(images))[-1]
      id_loss  = tf.math.reduce_mean(tf.math.abs(noise_repr - noise))
      id_loss += tf.math.reduce_mean(tf.math.abs(image_repr - images))

      d1_loss = real_loss - Kt * fake_loss
      g1_loss = fake_loss + id_weight * id_loss
      e1_loss = id_weight * id_loss

   g1_gradients = tape.gradient(g1_loss, g1.trainable_variables)
   g1_optimizer.apply_gradients(zip(g1_gradients, g1.trainable_variables))
   d1_gradients = tape.gradient(d1_loss, d1.trainable_variables)
   d1_optimizer.apply_gradients(zip(d1_gradients, d1.trainable_variables))
   e1_gradients = tape.gradient(e1_loss, e1.trainable_variables)
   e1_optimizer.apply_gradients(zip(e1_gradients, e1.trainable_variables))

   Kt = Kt + lambda_Kt * (gamma * real_loss - fake_loss)
   if Kt < 0.001:
      Kt = 0.001
   if Kt > 1.0:
      Kt = 1.0

   convergence = real_loss + tf.abs(gamma * real_loss - fake_loss)

   return real_loss, fake_loss, id_loss, Kt, convergence


def save_models():
   d1.save(SAVE_PATH+"/discriminator")
   g1.save(SAVE_PATH+"/generator1")
   if remote:
      print("Uploading model")
      subprocess.call([
    		'gsutil', 'cp', '-r',
			os.path.join(SAVE_PATH),
    		os.path.join('gs://', GCP_BUCKET)
      ])

# train the discriminator and decoder
Kt = 0
s=0
# manually enumerate epochs
while s < samples:
   j=0
   for element in dataset:
      s+=1
      j+=1
      if s%save_every == 0:
         save_models()
         print("saved")
      if s%lr_update_step == lr_update_step-1:
         if learning_rate > min_learning_rate:
            learning_rate = learning_rate/2
            tf_lr.assign(learning_rate)
      this_batch_size = element[0].shape[0]
      real_loss, fake_loss, id_loss, Kt, convergence = train(element[0], this_batch_size, Kt)
      if s%log_step == log_step-1:
         print(' %d, %d/%d, r1=%.3f, g=%.3f, e=%.3f, Kt=%.3f, convergence=%.3f' %
            (s, j, n_batches, real_loss, fake_loss, id_loss, Kt, convergence))

      if s > samples:
         break


save_models()
