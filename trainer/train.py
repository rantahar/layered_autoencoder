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

MODEL_PATH = 'began'
GCP_BUCKET = "rantahar-nn"
learning_rate = 0.001
gamma = 0.3
lambda_Kt = learning_rate
beta = 0.5
BATCH_SIZE = 32
IMG_SIZE = 64
critic_updates = 5
dcl = 64
gcl = 64
latent_dim = 128
gp_weight = 10

# Specific training parameters
epochs = 5
SAVE_PATH = MODEL_PATH # for local
DATA_PATH = 'celeba'
save_every = 3500

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
   save_every = 100
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




def make_decoder():
   in_shape=(latent_dim,)
   init = RandomNormal(stddev=0.02)
   model = Sequential()
   n_nodes = gcl * 8 * 4 * 4
   model.add(layers.Dense(n_nodes, input_shape=in_shape))
   model.add(layers.LeakyReLU(alpha=0.2))
   model.add(layers.Reshape((4, 4, gcl * 8)))
   model.add(layers.Conv2DTranspose(gcl*4, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
   model.add(layers.LeakyReLU(alpha=0.2))
   model.add(layers.Conv2DTranspose(gcl*2, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
   model.add(layers.LeakyReLU(alpha=0.2))
   model.add(layers.Conv2DTranspose(gcl, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
   model.add(layers.LeakyReLU(alpha=0.2))
   model.add(layers.Conv2DTranspose(3, (4,4), strides=(2,2), activation='tanh', padding='same', kernel_initializer=init))
   return model


def make_encoder():
   in_shape=(IMG_SIZE,IMG_SIZE,3)
   init = RandomNormal(stddev=0.02)
   model = Sequential()
   # normal
   model.add(layers.Conv2D(dcl, (4,4), strides=(2,2), padding='same', kernel_initializer=init, input_shape=in_shape))
   model.add(layers.LeakyReLU(alpha=0.2))
   model.add(layers.Conv2D(dcl*2, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
   model.add(layers.LeakyReLU(alpha=0.2))
   model.add(layers.Conv2D(dcl*4, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
   model.add(layers.LeakyReLU(alpha=0.2))
   model.add(layers.Conv2D(dcl*8, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
   model.add(layers.LeakyReLU(alpha=0.2))
   model.add(layers.Flatten())
   model.add(layers.Dense(latent_dim, activation='tanh',))
   return model

decoder = make_decoder()
decoder.summary()
encoder = make_encoder()
encoder.summary()
generator = make_decoder()
generator.summary()

def discriminator(image):
	x = encoder(image)
	x = decoder(x)
	return x



encoder_optimizer = tf.keras.optimizers.Adam(lr=learning_rate, beta_1=beta)
decoder_optimizer = tf.keras.optimizers.Adam(lr=learning_rate, beta_1=beta)
generator_optimizer = tf.keras.optimizers.Adam(lr=learning_rate, beta_1=beta)




@tf.function
def train(images, batch_size, Kt):
   noise = tf.random.normal([batch_size, latent_dim])

   with tf.GradientTape() as enc_tape, tf.GradientTape() as dec_tape, tf.GradientTape() as gen_tape:
      fake_images = generator(noise)
      real_quality = discriminator(images)
      fake_quality = discriminator(fake_images)

      real_loss = tf.math.reduce_mean(tf.math.abs(real_quality - images))
      fake_loss = tf.math.reduce_mean(tf.math.abs(fake_quality - fake_images))

      d_loss  = real_loss - Kt * fake_loss
      g_loss  = fake_loss

   generator_gradients = gen_tape.gradient(g_loss, generator.trainable_variables)
   generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
   encoder_gradients = enc_tape.gradient(d_loss, encoder.trainable_variables)
   encoder_optimizer.apply_gradients(zip(encoder_gradients, encoder.trainable_variables))
   decoder_gradients = dec_tape.gradient(d_loss, decoder.trainable_variables)
   decoder_optimizer.apply_gradients(zip(decoder_gradients, decoder.trainable_variables))

   Kt = Kt + lambda_Kt * (gamma * real_loss - fake_loss)
   if Kt < 0.0:
      Kt = 0.0
   if Kt > 1.0:
      Kt = 1.0

   convergence = real_loss + tf.abs(gamma * real_loss - fake_loss)

   return fake_loss, real_loss, Kt, convergence


def save_models():
   decoder.save(SAVE_PATH+"/decoder")
   encoder.save(SAVE_PATH+"/encoder")
   generator.save(SAVE_PATH+"/generator")
   if remote:
      print("Uploading model")
      subprocess.call([
    		'gsutil', 'cp', '-r',
			os.path.join(SAVE_PATH),
    		os.path.join('gs://', GCP_BUCKET)
      ])

# train the encoder and decoder
n_batches = tf.data.experimental.cardinality(dataset)
Kt = 0
# manually enumerate epochs
for i in range(epochs):
   for j, element in enumerate(dataset):
      if j%save_every == 0:
         save_models()
         print("saved")
      this_batch_size = element[0].shape[0]
      fake_loss, real_loss, Kt, convergence = train(element[0], this_batch_size, Kt)
      print('> %d, %d/%d, f=%.3f, r=%.3f, Kt=%.3f, convergence=%.3f' %
         (i+1, j+1, n_batches, fake_loss, real_loss, Kt, convergence))


save_models()
