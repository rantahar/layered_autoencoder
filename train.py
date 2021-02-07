import matplotlib.pyplot as plt
import numpy as np
import os
import datetime

import tensorflow as tf
import tensorflow_cloud as tfc
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.initializers import RandomNormal

from tensorflow.keras.preprocessing import image_dataset_from_directory

MODEL_PATH = 'model'
GCP_BUCKET = "rantahar-nn"
PATH = '../data/celeba'
learning_rate = 0.001
beta = 0.5
BATCH_SIZE = 32
IMG_SIZE = 64
enc_image_size = 16
encoding_dimension = 5
id_weight = 10
full_disc_weight = 0.1
dcl = 64
gcl = 64
latent_dim = 100

dataset = image_dataset_from_directory(PATH,
                                       shuffle=True,
                                       batch_size=BATCH_SIZE,
                                       image_size=(IMG_SIZE,IMG_SIZE))

print('Number of train batches: %d' % tf.data.experimental.cardinality(dataset))
print(dataset.class_names)

def normalize(image, label):
   image = tf.cast(image, tf.float32)
   image = (image / 127.5) - 1
   return image, label

dataset = dataset.map(normalize)

AUTOTUNE = tf.data.AUTOTUNE
dataset = dataset.prefetch(buffer_size=AUTOTUNE)





class ClipConstraint(Constraint):
	# set clip value when initialized
	def __init__(self, clip_value):
		self.clip_value = clip_value

	# clip model weights to hypercube
	def __call__(self, weights):
		return backend.clip(weights, -self.clip_value, self.clip_value)

	# get the config
	def get_config(self):
		return {'clip_value': self.clip_value}


def make_encoding_discriminator():
   in_shape=(enc_image_size,enc_image_size,encoding_dimension)
   #const = ClipConstraint(0.01)
   #init = RandomNormal(stddev=0.02)
   model = Sequential()
   # normal
   #model.add(layers.Conv2D(dcl, (4,4), padding='same', kernel_initializer=init,  kernel_constraint=const, input_shape=in_shape))
   model.add(layers.Conv2D(dcl, (4,4), padding='same', input_shape=in_shape))
   model.add(layers.LeakyReLU(alpha=0.2))
   # downsample
   model.add(layers.Conv2D(dcl, (4,4), strides=(2,2), padding='same'))
   model.add(layers.LeakyReLU(alpha=0.2))
   # downsample
   model.add(layers.Conv2D(dcl, (4,4), strides=(2,2), padding='same'))
   model.add(layers.LeakyReLU(alpha=0.2))
   # classifier
   model.add(layers.Flatten())
   model.add(layers.Dropout(0.4))
   model.add(layers.Dense(1))
   # compile model
   return model

def make_encoding_generator():
	model = Sequential()
	# foundation for 4x4 image
	n_nodes = gcl * 4 * 4 * 4
	model.add(layers.Dense(n_nodes, input_dim=latent_dim))
	model.add(layers.LeakyReLU(alpha=0.2))
	model.add(layers.Reshape((4, 4, gcl * 4)))
	# upsample to 8x8
	model.add(layers.Conv2DTranspose(gcl, (4,4), strides=(2,2), padding='same'))
	model.add(layers.LeakyReLU(alpha=0.2))
	# upsample to 16x16
	model.add(layers.Conv2DTranspose(gcl, (4,4), strides=(2,2), padding='same'))
	model.add(layers.LeakyReLU(alpha=0.2))
	model.add(layers.Conv2DTranspose(encoding_dimension, (4,4), strides=(1,1), activation='tanh', padding='same'))
	return model

# Generates a color image upsampled once
def make_decoder():
   in_shape=(enc_image_size,enc_image_size,encoding_dimension)
   model = Sequential()
   # input is a 32x32 image
   # upsample to 64x64
   model.add(layers.Conv2DTranspose(gcl, (4,4), strides=(2,2), padding='same', input_shape=in_shape))
   model.add(layers.LeakyReLU(alpha=0.2))
   model.add(layers.Conv2DTranspose(gcl, (4,4), strides=(2,2), padding='same'))
   model.add(layers.LeakyReLU(alpha=0.2))
   model.add(layers.Conv2D(3, (4,4), strides=(1,1), activation='tanh', padding='same'))
   return model

# Generates a downsampled image
def make_encoder():
   in_shape=(IMG_SIZE,IMG_SIZE,3)
   inputs = tf.keras.Input(shape=in_shape)
   x = inputs
   x = layers.Conv2D(gcl, (4,4), strides=(2,2), padding='same', input_shape=in_shape)(x)
   x = layers.LeakyReLU(alpha=0.2)(x)
   x = layers.Conv2D(gcl, (4,4), strides=(2,2), padding='same')(x)
   x = layers.LeakyReLU(alpha=0.2)(x)
   encoding = layers.Conv2D(encoding_dimension, (4,4), strides=(1,1), activation='sigmoid', padding='same')(x)

   model = tf.keras.Model(inputs=inputs, outputs=encoding, name="encoder")
   return model


discriminator = make_encoding_discriminator()
discriminator.summary()
generator = make_encoding_generator()
generator.summary()
encoder = make_encoder()
encoder.summary()
decoder = make_decoder()
decoder.summary()


encoder_optimizer = Adam(lr=learning_rate, beta_1=beta)
decoder_optimizer = Adam(lr=learning_rate, beta_1=beta)
discriminator_optimizer = Adam(lr=learning_rate, beta_1=beta)
generator_optimizer = Adam(lr=learning_rate, beta_1=beta)




cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output)-0.05, fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output)-0.05, real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output)+0.05, fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def Wasserstein_loss(real_output, fake_output):
   loss = tf.math.reduce_mean(real_output - fake_output)
   return loss

def generator_Wasserstein_loss(fake_output):
   loss = tf.math.reduce_mean(fake_output)
   return loss

def id_loss(original, generated):
   diff = original - generated
   return tf.math.reduce_mean(diff*diff)



@tf.function
def train_step(images, batch_size):
   noise = tf.random.normal([batch_size, latent_dim])

   with tf.GradientTape() as en_tape, tf.GradientTape() as dec_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
      encodings = encoder(images)
      decodings = decoder(encodings)
      id_loss_1 = id_loss(images, decodings)
      encodings_2 = encoder(decodings)
      id_loss_2 = id_loss(encodings, encodings_2)

      generated_encodings = generator(noise)
      real_quality_2 = discriminator(encodings)
      fake_quality_2 = discriminator(generated_encodings)

      d_loss = discriminator_loss(real_quality_2, fake_quality_2)
      g_loss_1 = generator_loss(fake_quality_2)

      decodings = decoder(generated_encodings)
      encodings = encoder(decodings)

      enc_loss = id_loss_1 + id_loss_2
      dec_loss = id_loss_1 + id_loss_2
      gen_loss =  g_loss_1
      disc_loss = d_loss

   encoder_gradients = en_tape.gradient(enc_loss, encoder.trainable_variables)
   decoder_gradients = dec_tape.gradient(dec_loss, decoder.trainable_variables)
   discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
   generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)

   encoder_optimizer.apply_gradients(zip(encoder_gradients, encoder.trainable_variables))
   decoder_optimizer.apply_gradients(zip(decoder_gradients, decoder.trainable_variables))
   discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))
   generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))

   return id_loss_1, id_loss_2, g_loss_1, d_loss





if tfc.remote():
   epochs = 500
   SAVE_PATH = os.path.join("gs://", GCP_BUCKET, MODEL_PATH)
   train_data = dataset
else:
   SAVE_PATH = MODEL_PATH
   epochs = 100
   train_data = dataset
   callbacks = None


# train the encoder and decoder
n_batches = tf.data.experimental.cardinality(dataset)
# manually enumerate epochs
for i in range(epochs):
   for j, element in enumerate(dataset):
      if j%100 == 0:
         encoder.save(SAVE_PATH+"/encoder")
         decoder.save(SAVE_PATH+"/decoder")
         generator.save(SAVE_PATH+"/generator")
         print("saved")
      this_batch_size = element[0].shape[0]
      id_loss_1, id_loss_2, g_loss_1, d_loss = train_step(element[0], this_batch_size)
      # summarize loss on this batch
      print('> %d, %d/%d, id1=%.3f, id2=%.3f, g=%.3f, d=%.3f' %
         (i+1, j+1, n_batches, id_loss_1, id_loss_2, g_loss_1, d_loss))


encoder.save(SAVE_PATH+"/encoder")
decoder.save(SAVE_PATH+"/decoder")
generator.save(SAVE_PATH+"/generator")
