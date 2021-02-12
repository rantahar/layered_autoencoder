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

MODEL_PATH = 'edbegan_8_8'
GCP_BUCKET = "rantahar-nn"
learning_rate = 0.0008
min_learning_rate = 0.00002
lr_update_step = 10000
gamma = 0.5
lambda_Kt = 0.001
reproduction_weight = 0.1
beta = 0.5
BATCH_SIZE = 16
IMG_SIZE = 64
dcl = 8
gcl = 8
sketch_dim = 8
latent_dim = 64

# Specific training parameters
epochs = 50
SAVE_PATH = MODEL_PATH # for local
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
   save_every = 500
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

def conv_block(x, size):
   x = layers.Conv2D(size, (4,4), padding='same', kernel_initializer=init)(x)
   x = layers.BatchNormalization()(x)
   x = layers.LeakyReLU(alpha=0.2)(x)
   #x = layers.Conv2D(size, (4,4), padding='same', kernel_initializer=init)(x)
   #x = layers.BatchNormalization()(x)
   #x = layers.LeakyReLU(alpha=0.2)(x)
   return x

def upscale_block(x, size, upscale = True):
   x = conv_block(x, size)
   if upscale:
      img_size = x.shape[1]
      x = tf.image.resize(x, (2*img_size, 2*img_size), method="nearest")
   return x

def downscale_block(x, size, downscale = True):
   x = conv_block(x, size)
   if downscale:
      x = layers.Conv2D(size, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(x)
      x = layers.BatchNormalization()(x)
      x = layers.LeakyReLU(alpha=0.2)(x)
   return x


def make_generator_step(input_shape, gcl, n_out):
   input = tf.keras.Input(shape=input_shape)
   x = upscale_block(input, gcl)
   x = upscale_block(x, gcl)
   x = upscale_block(x, gcl)
   output = layers.Conv2D(n_out, (4,4), activation='tanh', padding='same', kernel_initializer=init)(x)
   model = Model(inputs = input, outputs = output)
   return model

def make_first_generator(n_in, gcl, n_out):
   input = tf.keras.Input(shape=(n_in))
   n_nodes = gcl * 4 * 4
   x = layers.Dense(n_nodes)(input)
   x = layers.Reshape((4, 4, gcl))(x)
   x = upscale_block(x, gcl)
   output = layers.Conv2D(n_out, (4,4), activation='tanh', padding='same', kernel_initializer=init)(x)
   model = Model(inputs = input, outputs = output)
   return model

def combine_models(generators):
   input_shape = generators[0].input_shape[1:]
   input = tf.keras.Input(shape=input_shape)
   x = input
   for g in generators:
      x = g(x)
   output = x
   model = Model(inputs = input, outputs = output)
   return model

g1 = make_first_generator(latent_dim, gcl, sketch_dim)
g2 = make_generator_step((8, 8, sketch_dim), gcl, 3)
full_generator = combine_models((g1,g2))

g1.summary()
g2.summary()


def make_encoder(input_size, features, gcl, n_out):
   input = tf.keras.Input(shape=(input_size,input_size,features))
   # encoding
   x = downscale_block(input, gcl)
   x = downscale_block(x, gcl*2)
   x = downscale_block(x, gcl*4)
   x = conv_block(x, gcl*8)
   output = layers.Conv2DTranspose(n_out, (4,4), activation='tanh', padding='same', kernel_initializer=init)(x)
   model = Model(inputs = input, outputs = output)
   return model


sketcher = make_encoder(IMG_SIZE, 3, gcl, sketch_dim)
sketcher.summary()



def make_discriminator(input_shape, dcl, latent_dim):
   features = input_shape[-1]
   input_size = input_shape[1]
   input = tf.keras.Input(shape=input_shape)
   # encoding
   x = input
   x = downscale_block(input, dcl)
   x = downscale_block(x, dcl*2)
   x = downscale_block(x, dcl*4)
   x = downscale_block(x, dcl*8)
   x = conv_block(x, dcl*16)
   x = conv_block(x, dcl*16)
   x = conv_block(x, dcl*16)
   x = upscale_block(x, dcl)
   x = upscale_block(x, dcl)
   x = upscale_block(x, dcl)
   x = upscale_block(x, dcl)
   output = layers.Conv2DTranspose(features, (4,4), activation='tanh', padding='same', kernel_initializer=init)(x)
   model = Model(inputs = input, outputs = output)
   return model


d1 = make_discriminator(g2.output_shape[1:], dcl, latent_dim)
d1.summary()


tf_lr = tf.Variable(learning_rate)
d1_optimizer = tf.keras.optimizers.Adam(lr=tf_lr, beta_1=beta)
g1_optimizer = tf.keras.optimizers.Adam(lr=tf_lr, beta_1=beta)
g2_optimizer = tf.keras.optimizers.Adam(lr=tf_lr, beta_1=beta)
sketcher_optimizer = tf.keras.optimizers.Adam(lr=tf_lr, beta_1=beta)




@tf.function
def train(images, batch_size, Kt):
   noise = tf.random.uniform([batch_size, latent_dim], minval=-1)

   with tf.GradientTape(persistent=True) as tape:
      fake_images = full_generator(noise)
      reproduction = g2(sketcher(images))

      real_image_quality = d1(images)
      fake_image_quality = d1(fake_images)

      real_loss = tf.math.reduce_mean(tf.math.abs(real_image_quality - images))
      fake_loss = tf.math.reduce_mean(tf.math.abs(fake_image_quality - fake_images))
      reproduction_loss = tf.math.reduce_mean(tf.math.abs(reproduction - images))

      d1_loss = real_loss - Kt * fake_loss
      g2_loss = fake_loss + 0.1*reproduction_loss
      g1_loss = fake_loss
      sketcher_loss = reproduction_loss

   g1_gradients = tape.gradient(g1_loss, g1.trainable_variables)
   g1_optimizer.apply_gradients(zip(g1_gradients, g1.trainable_variables))
   g2_gradients = tape.gradient(g2_loss, g2.trainable_variables)
   g2_optimizer.apply_gradients(zip(g2_gradients, g2.trainable_variables))
   sketcher_gradients = tape.gradient(sketcher_loss, sketcher.trainable_variables)
   sketcher_optimizer.apply_gradients(zip(sketcher_gradients, sketcher.trainable_variables))
   d1_gradients = tape.gradient(d1_loss, d1.trainable_variables)
   d1_optimizer.apply_gradients(zip(d1_gradients, d1.trainable_variables))


   Kt = Kt + lambda_Kt * (gamma * real_loss - fake_loss)
   if Kt < 0.0:
      Kt = 0.0
   if Kt > 1.0:
      Kt = 1.0

   convergence = real_loss + tf.abs(gamma * real_loss - fake_loss)

   return real_loss, fake_loss, reproduction_loss, Kt, convergence


def save_models():
   d1.save(SAVE_PATH+"/discriminator")
   sketcher.save(SAVE_PATH+"/sketcher")
   g1.save(SAVE_PATH+"/generator1")
   g2.save(SAVE_PATH+"/generator2")
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
      real_loss, fake_loss, reproduction_loss, Kt, convergence = train(element[0], this_batch_size, Kt)
      if s%log_step == log_step-1:
         print(' %d, %d/%d, r1=%.3f, g=%.3f, g2=%.3f, Kt=%.3f, convergence=%.3f' %
            (s, j+1, n_batches, real_loss, fake_loss, reproduction_loss, Kt, convergence))


save_models()
