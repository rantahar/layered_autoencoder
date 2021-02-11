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
dcl = 16
gcl = 16
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

def upscale_block(x, size, upscale = True):
   x = layers.Conv2D(size, (4,4), padding='same', kernel_initializer=init)(x)
   x = layers.BatchNormalization()(x)
   x = layers.LeakyReLU(alpha=0.2)(x)
   #x = layers.Conv2D(size, (4,4), padding='same', kernel_initializer=init)(x)
   #x = layers.BatchNormalization()(x)
   #x = layers.LeakyReLU(alpha=0.2)(x)
   if upscale:
      img_size = x.shape[1]
      x = tf.image.resize(x, (2*img_size, 2*img_size), method="nearest")
   return x

def downscale_block(x, size, downscale = True):
   x = layers.Conv2D(size, (4,4), padding='same', kernel_initializer=init)(x)
   x = layers.BatchNormalization()(x)
   x = layers.LeakyReLU(alpha=0.2)(x)
   #x = layers.Conv2D(size, (4,4), padding='same', kernel_initializer=init)(x)
   #x = layers.BatchNormalization()(x)
   #x = layers.LeakyReLU(alpha=0.2)(x)
   if downscale:
      x = layers.Conv2D(size, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(x)
      x = layers.BatchNormalization()(x)
      x = layers.LeakyReLU(alpha=0.2)(x)
   return x


n_steps = 0
size = 4
while size < IMG_SIZE:
   size*=2
   n_steps += 1

def make_encoder_step(input_shape, n_in, dcl):
   input = tf.keras.Input(shape=(*input_shape, n_in))
   x = downscale_block(input, dcl)
   x = downscale_block(x, dcl)
   output = downscale_block(x, dcl)
   model = Model(inputs = input, outputs = output)
   return model

def make_full_encoder(encoder, dcl, n_out):
   input_shape = encoder.input_shape
   input = tf.keras.Input(shape=input_shape[1:])
   e1 = encoder(input)
   x = downscale_block(e1, dcl)
   x = layers.Flatten()(x)
   x = layers.Dense(2*n_out)(x)
   x = layers.LeakyReLU(alpha=0.2)(x)
   e2 = layers.Dense(n_out, activation='tanh')(x)
   model = Model(inputs = input, outputs = [e1,e2])
   return model


def make_generator_step(input_shape, gcl, n_out, image=False):
   input = tf.keras.Input(shape=(*input_shape, gcl))
   x = upscale_block(input, gcl)
   x = upscale_block(x, gcl)
   output = upscale_block(x, gcl)
   if image:
      output = layers.Conv2DTranspose(n_out, (4,4), activation='tanh', padding='same', kernel_initializer=init)(output)
   model = Model(inputs = input, outputs = output)
   return model

def make_full_generator(n_in, gcl, generator_head):
   input = tf.keras.Input(shape=(n_in))
   n_nodes = gcl * 4 * 4
   x = layers.Dense(n_nodes)(input)
   x = layers.Reshape((4, 4, gcl))(x)
   x = upscale_block(x, gcl)
   output = generator_head(x)
   model = Model(inputs = input, outputs = output)
   return model


e1 = make_encoder_step((64, 64), 3, gcl)
g1 = make_generator_step((8, 8), gcl, 3, image=True)

e2 = make_full_encoder(e1, gcl, latent_dim)
g2 = make_full_generator(latent_dim, gcl, g1)

encoders = [e1, e2]
generators = [g1, g2]
full_encoder = e2
full_generator = g2

for e in encoders:
   e.summary()

for g in generators:
   g.summary()



def make_discriminator():
   input = tf.keras.Input(shape=(IMG_SIZE,IMG_SIZE,3))
   # encoding
   x = input
   s = 1
   for i in range(4):
      x = downscale_block(x, dcl*s)
      s*=2
   x = downscale_block(x,  dcl*s, downscale = False)
   x = layers.Flatten()(x)
   x = layers.Dense(latent_dim, activation='tanh')(x)

   # decoding
   n_nodes = gcl * 4 * 4
   x = layers.Flatten()(x)
   x = layers.Dense(n_nodes)(x)
   x = layers.LeakyReLU(alpha=0.2)(x)
   x = layers.Reshape((4, 4, gcl))(x)
   for i in range(4):
      x = upscale_block(x, gcl)
   x = upscale_block(x, gcl, upscale = False)
   output = layers.Conv2DTranspose(3, (4,4), activation='tanh', padding='same', kernel_initializer=init)(x)
   model = Model(inputs = input, outputs = output)
   return model

discriminator = make_discriminator()
discriminator.summary()


tf_lr = tf.Variable(learning_rate)
discriminator_optimizer = tf.keras.optimizers.Adam(lr=tf_lr, beta_1=beta)
generator_optimizer = tf.keras.optimizers.Adam(lr=tf_lr, beta_1=beta)
encoder_optimizer = tf.keras.optimizers.Adam(lr=tf_lr, beta_1=beta)




@tf.function
def train(images, batch_size, Kt):
   noise = tf.random.uniform([batch_size, latent_dim], minval=-1)

   with tf.GradientTape(persistent=True) as tape:
      fake_images = full_generator(noise)
      real_quality = discriminator(images)
      fake_quality = discriminator(fake_images)

      encodings = full_encoder(images)
      reproduction_loss = 0
      for e, g in zip(encodings, generators):
         reproduction = g(e)
         reproduction_loss += tf.math.reduce_mean(tf.math.square(reproduction - images))

      real_loss = tf.math.reduce_mean(tf.math.square(real_quality - images))
      fake_loss = tf.math.reduce_mean(tf.math.square(fake_quality - fake_images))

      d_loss = real_loss - Kt * fake_loss
      e_loss = reproduction_weight * reproduction_loss
      g_loss = fake_loss + e_loss

   generator_gradients = tape.gradient(g_loss, full_generator.trainable_variables)
   generator_optimizer.apply_gradients(zip(generator_gradients, full_generator.trainable_variables))
   encoder_gradients = tape.gradient(e_loss, full_encoder.trainable_variables)
   encoder_optimizer.apply_gradients(zip(encoder_gradients, full_encoder.trainable_variables))
   discriminator_gradients = tape.gradient(d_loss, discriminator.trainable_variables)
   discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

   Kt = Kt + lambda_Kt * (gamma * real_loss - fake_loss)
   if Kt < 0.0:
      Kt = 0.0
   if Kt > 1.0:
      Kt = 1.0

   convergence = real_loss + tf.abs(gamma * real_loss - fake_loss)

   return fake_loss, reproduction_loss, real_loss, Kt, convergence


def save_models():
   discriminator.save(SAVE_PATH+"/discriminator")
   full_generator.save(SAVE_PATH+"/generator")
   encoders[0].save(SAVE_PATH+"/encoder")
   generators[0].save(SAVE_PATH+"/decoder")
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
avrp, avre, avf, avc = 1, 1, 1, 1
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
         avrp = 0.95*avrp + 0.05*repr_loss
         avre = 0.95*avre + 0.05*real_loss
         avf = 0.95*avf + 0.05*fake_loss
         avc = 0.95*avc + 0.05*convergence
         print(' %d, %d/%d, f=%.3f, repr=%.3f, r=%.3f, Kt=%.3f, convergence=%.3f' %
            (s, j+1, n_batches, avf, avrp, avre, Kt, avc))


save_models()
