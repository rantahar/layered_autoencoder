# Creates a single image size independent autoencoder block and
# trains it on the celeb_a or the imagenet dataset
import sys
import tensorflow as tf
from stackable_autoencoder.models import Autoencoder
import stackable_autoencoder.data


BATCH_SIZE = 16
IMG_SIZE = 64
size = 32
encoding_size = 32
scalings = 3
learning_rate = 0.00005
#train_data_path = "OpenImagesDataset256/train"
#valid_data_path = "OpenImagesDataset256/valid"
train_data_path = "celeba"
valid_data_path = "celeba"
samples = 100000

remote = False
save_every = 50000
log_step = 1
train_data_path = "../data/"+train_data_path
valid_data_path = "../data/"+valid_data_path


train_dataset = stackable_autoencoder.data.dataset_from_folder(train_data_path, IMG_SIZE, BATCH_SIZE)

n_batches = tf.data.experimental.cardinality(train_dataset)
epochs = samples//n_batches + 1

valid_dataset = stackable_autoencoder.data.dataset_from_folder(valid_data_path, IMG_SIZE, BATCH_SIZE)

autoencoder = Autoencoder(size=size, n_out=encoding_size, n_scalings = scalings)
#autoencoder = Autoencoder(size=size, n_out=encoding_size, n_scalings = scalings, load=True)

train_dataset = train_dataset.prefetch(100)
valid_dataset = valid_dataset.prefetch(100)
autoencoder.train(train_dataset, valid_dataset = train_dataset, epochs=epochs, save_every = save_every, learning_rate = learning_rate)
autoencoder.save()
