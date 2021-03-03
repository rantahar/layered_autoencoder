import tensorflow as tf
from stackable_autoencoder.models import BlockedAutoencoder
import stackable_autoencoder.data


BATCH_SIZE = 16
IMG_SIZE = 64
size = 64
encoding_size = 64
latent_dim = 64
scalings = 3
samples = 10000


if len(sys.argv) > 2 and sys.argv[1] == "remote":
   remote = True
   save_every = 5000
   log_step = 1
   bucket = sys.argv[1]
else:
   remote = False
   save_every = 500
   log_step = 1
   bucket = None



train_dataset, valid_dataset = stackable_autoencoder.data.get_celeba(IMG_SIZE, BATCH_SIZE)
n_batches = tf.data.experimental.cardinality(train_dataset)
epochs = samples//n_batches + 1

autoencoder = BlockedAutoencoder(IMG_SIZE, size, encoding_size, latent_dim,
                                 scalings_per_step = scalings)

#train_dataset = train_dataset.take(100)
#valid_dataset = valid_dataset.take(10)
print_log()
autoencoder.train(train_dataset, valid_dataset, epochs, bucket = bucket, log_step = log_step, save_every = save_every)
autoencoder.save(bucket = bucket)
