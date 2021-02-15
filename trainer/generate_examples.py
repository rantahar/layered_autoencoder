import sys
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image_dataset_from_directory


if len(sys.argv) > 1:
   MODEL_PATH = sys.argv[1]
else:
   MODEL_PATH = "began"


AUTOENCODER_PATH = f'autoencoder_64_32_8'


latent_dim = 64
PATH = '../data/celeba'
IMG_SIZE = 64
BATCH_SIZE = 5

def normalize(image, label):
   image = tf.cast(image, tf.float32)
   image = (image / 127.5) - 1
   return image, label

dataset = image_dataset_from_directory(PATH,
                                       shuffle=True,
                                       batch_size=BATCH_SIZE,
                                       image_size=(IMG_SIZE,IMG_SIZE))

dataset = dataset.take(1).map(normalize)
images = next(iter(dataset))[0]


noise = tf.random.uniform([16, latent_dim], minval=-1)
discriminator = tf.keras.models.load_model(MODEL_PATH+"/discriminator")
generator = tf.keras.models.load_model(MODEL_PATH+"/generator")
encoder = tf.keras.models.load_model(AUTOENCODER_PATH+"/encoder")
decoder = tf.keras.models.load_model(AUTOENCODER_PATH+"/decoder")

reproduced = decoder(encoder(images))
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
    fig.add_subplot(4, 4, 4*i+3)
    im = (discriminated[i] + 1) / 2
    plt.imshow(im)
    fig.add_subplot(4, 4, 4*i+4)
    im = (generated[i] + 1) / 2
    plt.imshow(im)


for ax in fig.axes:
    ax.axis("off")

plt.show()
