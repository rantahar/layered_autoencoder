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
encoder1 = tf.keras.models.load_model(MODEL_PATH+"/encoder1")
decoder1 = tf.keras.models.load_model(MODEL_PATH+"/decoder1")
discriminator2 = tf.keras.models.load_model(MODEL_PATH+"/discriminator2")
generator1 = tf.keras.models.load_model(MODEL_PATH+"/generator1")
generator2 = tf.keras.models.load_model(MODEL_PATH+"/generator2")

encoded = encoder1(images)
discriminated1 = decoder1(encoded)
discriminated2 = decoder1(discriminator2(encoded))
generated1 = generator2(encoded)
generated2 = generator2(generator1(noise))


fig=plt.figure(figsize=(8, 8))

for i in range(4):
    fig.add_subplot(4, 5, 5*i+1)
    im = (images[i] + 1) / 2
    plt.imshow(im)
    fig.add_subplot(4, 5, 5*i+2)
    im = (discriminated1[i] + 1) / 2
    plt.imshow(im)
    fig.add_subplot(4, 5, 5*i+3)
    im = (discriminated2[i] + 1) / 2
    plt.imshow(im)
    fig.add_subplot(4, 5, 5*i+4)
    im = (generated1[i] + 1) / 2
    plt.imshow(im)
    fig.add_subplot(4, 5, 5*i+5)
    im = (generated2[i] + 1) / 2
    plt.imshow(im)


for ax in fig.axes:
    ax.axis("off")

plt.show()
