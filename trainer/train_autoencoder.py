import numpy as np
import os
import subprocess

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras import backend
from tensorflow.keras.initializers import RandomNormal

from trainer.models import Autoencoder
import trainer.data


BATCH_SIZE = 16
IMG_SIZE = 64
size = 32
encoding_size = 32

# Specific training parameters
remote = False
samples = 10000
DATA_PATH = 'celeba'


if remote:
   save_every = 5000
   log_step = 50
   bucket = "rantahar-nn"
else:
   save_every = 500
   log_step = 1
   DATA_PATH = '../data/' + DATA_PATH
   bucket = None


#dataset = trainer.data.from_folder(DATA_PATH, IMG_SIZE, BATCH_SIZE, bucket)
dataset = trainer.data.get_celeba(IMG_SIZE, BATCH_SIZE)
n_batches = tf.data.experimental.cardinality(dataset)
epochs = samples//n_batches + 1


autoencoder = Autoencoder(IMG_SIZE, size, encoding_size)

if remote:
   autoencoder.train(dataset, epochs, bucket = GCP_BUCKET,
                     learning_rate = learning_rate, log_step = log_step,
                     save_every = save_every)
   autoencoder.save(bucket = GCP_BUCKET)

else:
   autoencoder.train(dataset, epochs, learning_rate = learning_rate,
                     log_step = log_step, save_every = save_every)
   autoencoder.save()
