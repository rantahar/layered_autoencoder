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
train_data_path = "gom/train"
valid_data_path = "gom/valid"
samples = 500000

if len(sys.argv) > 2 and sys.argv[1] == "remote":
   remote = True
   save_every = 5000
   log_step = 1
   bucket = sys.argv[2]
else:
   remote = False
   save_every = 50000
   log_step = 1
   train_data_path = "../data/"+train_data_path
   valid_data_path = "../data/"+valid_data_path
   bucket = None


train_dataset = stackable_autoencoder.data.dataset_from_folder(train_data_path, IMG_SIZE, BATCH_SIZE, bucket = bucket)
n_batches = tf.data.experimental.cardinality(train_dataset)
epochs = samples//n_batches + 1

valid_dataset = stackable_autoencoder.data.dataset_from_folder(valid_data_path, IMG_SIZE, BATCH_SIZE, bucket = bucket)

autoencoder = Autoencoder(size=size, n_out=encoding_size, n_scalings = scalings, bucket=bucket)

#train_dataset = train_dataset.take(1)
#valid_dataset = valid_dataset.take(1)
autoencoder.train(train_dataset, valid_dataset = train_dataset, epochs=epochs, bucket = bucket, save_every = save_every, learning_rate = learning_rate)
autoencoder.save(bucket)
