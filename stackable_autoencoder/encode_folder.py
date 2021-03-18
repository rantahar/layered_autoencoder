import numpy as np
import os
import sys

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras import backend
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.preprocessing import image

import stackable_autoencoder.data
from stackable_autoencoder import models

IMG_SIZE = 64


if len(sys.argv) > 1:
   autoencoder_path = sys.argv[1]
else:
    print("provide autoencoder path")
    sys.exit(1)

if len(sys.argv) > 2:
   data_path = sys.argv[2]
else:
   print("provide data path")
   sys.exit(1)

# Get the autoencoder
autoencoder = models.Autoencoder(load = True, save_path = autoencoder_path)

images = None
for label in os.scandir(data_path):
   n_files = len(os.listdir(label))
   batch = []
   for i, file in enumerate(os.scandir(label)):
      img = stackable_autoencoder.data.load_image(file)
      img = tf.image.resize(img, (IMG_SIZE,IMG_SIZE))
      img = stackable_autoencoder.data.normalize(img, 1)
      batch.append(img)
      if len(batch) >= 256:
         print(f"{i}/{n_files}")
         encoded, logvar = autoencoder.encode(tf.stack(batch))
         encoded = encoded.numpy()
         batch = []
         if images is None:
            images = encoded
         else:
            images = np.concatenate((images, encoded), 0)
         del encoded

result = np.array(images)
np.save("encoded.data", result)
