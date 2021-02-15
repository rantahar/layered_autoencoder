import sys
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image_dataset_from_directory
import trainer.data

if len(sys.argv) > 1:
   autoencoder_path = sys.argv[1]
else:
    print("provide autoencoder path")
    sys.exit(1)

if len(sys.argv) > 2:
   gan_path = sys.argv[2]
else:
    gan_path = None


latent_dim = 64
IMG_SIZE = 64
BATCH_SIZE = 5

dataset = trainer.data.get_celeba(IMG_SIZE, BATCH_SIZE)
images = next(iter(dataset.take(1)))


encoder = tf.keras.models.load_model(autoencoder_path+"/encoder")
decoder = tf.keras.models.load_model(autoencoder_path+"/decoder")
if gan_path is not None:
    discriminator = tf.keras.models.load_model(MODEL_PATH+"/discriminator")
    generator = tf.keras.models.load_model(MODEL_PATH+"/generator")

reproduced = decoder(encoder(images))

if gan_path is not None:
    noise = tf.random.uniform([16, latent_dim], minval=-1)
    discriminated = discriminator(images)
    generated = generator(noise)


fig=plt.figure(figsize=(8, 8))

for i in range(4):
    fig.add_subplot(4, 4, 4*i+1)
    im = (images[i] + 1) / 2
    plt.imshow(im)
    fig.add_subplot(4, 4, 4*i+2)
    im = (reproduced[i] + 1) / 2
    plt.imshow(im)
    if gan_path is not None:
        fig.add_subplot(4, 4, 4*i+3)
        im = (discriminated[i] + 1) / 2
        plt.imshow(im)
        fig.add_subplot(4, 4, 4*i+4)
        im = (generated[i] + 1) / 2
        plt.imshow(im)


for ax in fig.axes:
    ax.axis("off")

plt.show()
