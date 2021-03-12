import sys
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image_dataset_from_directory
from stackable_autoencoder.models import BlockedAutoencoder, Autoencoder
import stackable_autoencoder.data

if len(sys.argv) > 1:
   autoencoder_path = sys.argv[1]
else:
    print("provide autoencoder path")
    sys.exit(1)

if len(sys.argv) > 2:
   gan_path = sys.argv[2]
else:
    gan_path = None

#data_path = '../data/gom/train/'
data_path = '../data/celeba/'
IMG_SIZE = 64
BATCH_SIZE = 5

dataset = stackable_autoencoder.data.dataset_from_folder(data_path, IMG_SIZE, BATCH_SIZE)
images = next(iter(dataset.take(1)))

autoencoder = Autoencoder(save_path = autoencoder_path, load = True)
reproduced = autoencoder.call(images)

if gan_path is not None:
    discriminator = tf.keras.models.load_model(gan_path+"/discriminator")
    generator = tf.keras.models.load_model(gan_path+"/generator")

    latent_dim = generator.input_shape[-1]

    noise = tf.random.uniform([BATCH_SIZE, latent_dim], minval=-1)
    discriminated = discriminator(images)
    generated = generator(noise)


print("autoencoder loss: ", autoencoder.evaluate_batch(images).numpy())

fig=plt.figure(figsize=(8, 8))

n_im = 4

for i in range(BATCH_SIZE):
    fig.add_subplot(BATCH_SIZE, n_im, n_im*i+1)
    im = (images[i] + 1) / 2
    plt.imshow(im)
    fig.add_subplot(BATCH_SIZE, n_im, n_im*i+2)
    im = (reproduced[i] + 1) / 2
    plt.imshow(im)
    if gan_path is not None:
        fig.add_subplot(BATCH_SIZE, n_im, n_im*i+3)
        im = (discriminated[i] + 1) / 2
        plt.imshow(im)
        fig.add_subplot(BATCH_SIZE, n_im, n_im*i+4)
        im = (generated[i] + 1) / 2
        plt.imshow(im)


for ax in fig.axes:
    ax.axis("off")

plt.show()
