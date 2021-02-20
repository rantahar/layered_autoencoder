# Creates a single image size independent autoencoder block and
# trains it on the celeb_a or the imagenet dataset
import sys
import tensorflow as tf
from stackable_autoencoder.models import Autoencoder
import stackable_autoencoder.data


BATCH_SIZE = 16
IMG_SIZE = 64
size = 64
encoding_size = 64
scalings = 3
data_path = "celeba"

# Specific training parameters
samples = 50000

if len(sys.argv) > 2 and sys.argv[1] == "remote":
   remote = True
   save_every = 5000
   log_step = 50
   bucket = sys.argv[1]
else:
   remote = False
   save_every = 500
   log_step = 1
   data_path = "../data/celeba"
   bucket = None


train_dataset = stackable_autoencoder.data.dataset_from_folder(data_path, IMG_SIZE, BATCH_SIZE, bucket = bucket)
n_batches = tf.data.experimental.cardinality(train_dataset)
#train_dataset = stackable_autoencoder.data.list_from_folder(data_path)
#n_batches = len(train_dataset)
epochs = samples//n_batches + 1

autoencoder = Autoencoder(size=size, n_out=encoding_size, n_scalings = scalings)

#train_dataset = train_dataset.take(1)
#valid_dataset = valid_dataset.take(1)
autoencoder.train(train_dataset, epochs, n_batches, log_step = log_step, bucket = bucket, save_every = save_every)
#validation_loss = autoencoder.evaluate(valid_data)
autoencoder.save(bucket)
