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
learning_rate = 0.0001
min_learning_rate = 0.00002
lr_update_step = 10000
pure_autoencoder_updates = 1000
gamma = 0.5
lambda_Kt = 0.001
id_weight = 0.1
beta = 0.5
BATCH_SIZE = 16
IMG_SIZE = 64
dcl = 8
gcl = 8
latent_dim = 32
middle_latent_dim = 8

MODEL_PATH = f'edbegan_{IMG_SIZE}_{dcl}_{gcl}_{middle_latent_dim}'

# Specific training parameters
samples = 10000
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
   save_every = 500
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
                                       shuffle=True,
                                       batch_size=BATCH_SIZE,
                                       image_size=(IMG_SIZE,IMG_SIZE))

dataset = dataset.map(normalize).map(flip)
n_batches = tf.data.experimental.cardinality(dataset)
epochs = samples//n_batches + 1


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



def make_small_generator(n_in, gcl, n_out):
   input = tf.keras.Input(shape=(n_in))
   n_nodes = gcl * 4 * 4
   x = layers.Dense(n_nodes)(input)
   x = layers.Reshape((4, 4, gcl))(x)
   x = upscale_block(x, gcl)
   x = conv_block(x, gcl)
   output = layers.Conv2D(n_out, (4,4), activation='tanh', padding='same', kernel_initializer=init)(x)
   model = Model(inputs = input, outputs = output)
   return model

def make_big_generator(input_shape, gcl, n_out):
   input = tf.keras.Input(shape=input_shape)
   size = input_shape[1]
   x = input
   image = to_rpg(x)
   while size < IMG_SIZE:
      x = upscale_block(x, gcl)
      image = upscale(image) + to_rpg(x)
      size *= 2
   model = Model(inputs = input, outputs = image)
   return model

def make_small_encoder(input_shape, dcl, latent_dim):
   input = tf.keras.Input(shape=input_shape)
   x = downscale_block(input, dcl)
   x = conv_block(x, dcl)
   x = layers.Flatten()(x)
   x = layers.Dense(latent_dim*2)(x)
   output = layers.Dense(latent_dim)(x)
   model = Model(inputs = input, outputs = output)
   return model

def make_big_encoder(dcl, n_out):
   input = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
   size = IMG_SIZE
   s = 1
   x = input
   while size > 8:
      x = downscale_block(x, dcl)
      s*=2
      size /= 2
   x = conv_block(x, dcl)
   output = layers.Conv2D(n_out, (4,4), activation='tanh', padding='same', kernel_initializer=init)(x)
   model = Model(inputs = input, outputs = output)
   return model

def combine_models(steps):
   input_shape = steps[0].input_shape[1:]
   input = tf.keras.Input(shape=input_shape)
   x = input
   for s in steps:
      x =s(x)
   output = x
   model = Model(inputs = input, outputs = output)
   return model

# encoder: image to latend dim
big_enc = make_big_encoder(dcl, middle_latent_dim)
big_enc.summary()
#small_enc = make_small_encoder(big_enc.output_shape[1:], dcl, latent_dim)
#small_enc.summary()

# decoder: latent dim to image
#small_dec = make_small_generator(latent_dim, gcl)
#small_dec.summary()
big_dec = make_big_generator(big_enc.output_shape[1:], gcl, 3)
big_dec.summary()

# full autoencoder
autoencoder = combine_models((big_enc, big_dec))

# dicriminator: combine small encoder and decoder
d1 = make_small_encoder(big_enc.output_shape[1:], dcl, latent_dim)
d1.summary()
d2 = make_small_generator(latent_dim, gcl, middle_latent_dim)
d2.summary()

small_discriminator = combine_models((d1, d2))

# full dicscriminator
discriminator = combine_models((big_enc, small_discriminator, big_dec))

# generator: just a small decoder
small_generator = make_small_generator(latent_dim, gcl, middle_latent_dim)
small_generator.summary()

# full generator
generator = combine_models((small_generator, big_dec))

tf_lr = tf.Variable(learning_rate)
disc_optimizer = tf.keras.optimizers.Adam(lr=tf_lr, beta_1=beta)
gen_optimizer = tf.keras.optimizers.Adam(lr=tf_lr, beta_1=beta)
ae_optimizer = tf.keras.optimizers.Adam(lr=tf_lr, beta_1=beta)



@tf.function
def train_autoencoder(images):
   with tf.GradientTape(persistent=True) as tape:
      reproduction = autoencoder(images)
      loss = tf.math.reduce_mean(tf.math.abs(reproduction - images))

   gradients = tape.gradient(loss, autoencoder.trainable_variables)
   ae_optimizer.apply_gradients(zip(gradients, autoencoder.trainable_variables))
   return loss



@tf.function
def train_discriminator(images, batch_size, Kt):
   z = tf.random.uniform([batch_size, latent_dim], minval=-1)
   image_encodings = big_enc(images)

   with tf.GradientTape(persistent=True) as tape:
      real_quality = small_discriminator(image_encodings)
      real_loss = tf.math.reduce_mean(tf.math.abs(real_quality - image_encodings))

      fake_encodings = small_generator(z)
      fake_qualities = small_discriminator(fake_encodings)
      z_d = d1(fake_encodings)

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

   with tf.GradientTape(persistent=True) as tape:
      fake_encodings = small_generator(z)
      fake_qualities = small_discriminator(fake_encodings)
      loss = tf.math.reduce_mean(tf.math.abs(fake_encodings - fake_qualities))

   gradients = tape.gradient(loss, small_generator.trainable_variables)
   gen_optimizer.apply_gradients(zip(gradients, small_generator.trainable_variables))


def save_models():
   discriminator.save(SAVE_PATH+"/discriminator")
   generator.save(SAVE_PATH+"/generator")
   autoencoder.save(SAVE_PATH+"/autoencoder")
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

      ae_loss = train_autoencoder(element[0])

      if s < pure_autoencoder_updates:
         if s%log_step == log_step-1:
            print(' %d, %d/%d, ae=%.3f' % (s, j, n_batches, ae_loss))

      else:
         this_batch_size = element[0].shape[0]
         real_loss, fake_loss, id_loss, Kt, convergence =   train_discriminator(element[0], this_batch_size, Kt)
         train_generator()
         if s%log_step == log_step-1:
            print(' %d, %d/%d, ae=%.3f, r1=%.3f, g=%.3f, e=%.3f, Kt=%.3f, convergence=%.3f' %
               (s, j, n_batches, ae_loss, real_loss, fake_loss, id_loss, Kt, convergence))

print("DONE, saving...")
save_models()
print("saved")
