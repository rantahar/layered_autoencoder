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
size = 32
middle_latent_dim = 8

MODEL_PATH = f'autoencoder_{IMG_SIZE}_{size}_{middle_latent_dim}'

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


from trainer import models


# encoder: image to latend dim
encoder = models.make_big_encoder(size, IMG_SIZE, middle_latent_dim)
encoder.summary()

# decoder: latent dim to image
decoder = models.make_big_generator(encoder.output_shape[1:], size, IMG_SIZE, 3)
decoder.summary()

# full autoencoder
autoencoder = models.combine_models((encoder, decoder))

tf_lr = tf.Variable(learning_rate)
ae_optimizer = tf.keras.optimizers.Adam(lr=tf_lr, beta_1=beta)



@tf.function
def train_autoencoder(images):
   with tf.GradientTape(persistent=True) as tape:
      reproduction = autoencoder(images)
      loss = tf.math.reduce_mean(tf.math.abs(reproduction - images))

   gradients = tape.gradient(loss, autoencoder.trainable_variables)
   ae_optimizer.apply_gradients(zip(gradients, autoencoder.trainable_variables))
   return loss



def save_models():
   encoder.save(SAVE_PATH+"/encoder")
   decoder.save(SAVE_PATH+"/decoder")
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

      if s%log_step == log_step-1:
         print(' %d, %d/%d, ae=%.3f' % (s, j, n_batches, ae_loss))

print("DONE, saving...")
save_models()
print("saved")
