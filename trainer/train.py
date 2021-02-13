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
dcl = 8
gcl = 8
latent_dim = 64

MODEL_PATH = f'edbegan_{IMG_SIZE}_{dcl}_{gcl}'

# Specific training parameters
samples = 4000
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
   save_every = 1000
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
   image = to_rpg(x)
   size = 8
   while size < IMG_SIZE:
      x = upscale_block(x, gcl)
      image = upscale(image) + to_rpg(x)
      size *= 2
   model = Model(inputs = input, outputs = image)
   return model


g1 = make_generator(latent_dim, gcl, 3)
g1.summary()

def make_encoder(input_shape, dcl, latent_dim):
   features = input_shape[-1]
   input_size = input_shape[1]
   input = tf.keras.Input(shape=input_shape)
   x = input
   size = 64
   s = 1
   while size > 4:
      x = downscale_block(x, dcl*s)
      s*=2
      size /= 2
   x = downscale_block(x, dcl*s)
   x = layers.Flatten()(x)
   x = layers.Dense(latent_dim*2)(x)
   output = layers.Dense(latent_dim)(x)
   model = Model(inputs = input, outputs = output)
   return model

enc1 = make_encoder((IMG_SIZE, IMG_SIZE, 3), dcl, latent_dim)
enc1.summary()

dec1 = make_generator(latent_dim, gcl, 3)
dec1.summary()

def combine_models((steps)):
   input_shape = steps[0].input_shape[1:]
   input = tf.keras.Input(shape=input_shape)
   x = input
   for s in steps:
      x =s(x)
   output = x
   model = Model(inputs = input, outputs = output)
   return model

d1 = combine_models((enc1, dec1))
d1.summary()


tf_lr = tf.Variable(learning_rate)
d1_optimizer = tf.keras.optimizers.Adam(lr=tf_lr, beta_1=beta)
g1_optimizer = tf.keras.optimizers.Adam(lr=tf_lr, beta_1=beta)
ae_optimizer = tf.keras.optimizers.Adam(lr=tf_lr, beta_1=beta)




@tf.function
def train_discriminator(images, batch_size, Kt):
   z = tf.random.uniform([batch_size, latent_dim], minval=-1)

   with tf.GradientTape(persistent=True) as tape:
      real_image_quality = d1(images)
      real_loss = tf.math.reduce_mean(tf.math.abs(real_image_quality - images))

      fake_images = g1(z)
      fake_qualities = d1(fake_images)
      encoding = enc1(fake_images)

      fake_loss = tf.math.reduce_mean(tf.math.abs(fake_images - fake_qualities))
      id_loss = tf.math.reduce_mean(tf.math.abs(encoding - z))
      d1_loss = real_loss - Kt * fake_loss + id_weight * id_loss

   d1_gradients = tape.gradient(d1_loss, d1.trainable_variables)
   d1_optimizer.apply_gradients(zip(d1_gradients, d1.trainable_variables))

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

   with tf.GradientTape(persistent=True) as tape:
      fake_images = g1(z)
      fake_qualities = d1(fake_images)
      fake_loss = tf.math.reduce_mean(tf.math.abs(fake_images - fake_qualities))

   g1_gradients = tape.gradient(fake_loss, g1.trainable_variables)
   g1_optimizer.apply_gradients(zip(g1_gradients, g1.trainable_variables))


def save_models():
   d1.save(SAVE_PATH+"/discriminator")
   g1.save(SAVE_PATH+"/generator")
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
      real_loss, fake_loss, id_loss, Kt, convergence = train_discriminator(element[0], this_batch_size, Kt)
      train_generator()
      if s%log_step == log_step-1:
         print(' %d, %d/%d, r1=%.3f, g=%.3f, e=%.3f, Kt=%.3f, convergence=%.3f' %
            (s, j, n_batches, real_loss, fake_loss, id_loss, Kt, convergence))

      if s > samples:
         break


save_models()
