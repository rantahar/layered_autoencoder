
# Stackable autoencoder

The stackable variational autoencoder (VAE) encodes an image into a smaller
image with a given number of features per pixel. It is otherwise a standard
VAE, it consists of an encoder and a decoder, which are trained to
reproduce the original image from the encoding.

The idea is to speed up training a generative model (for example another VAE)
by producing only the encoded state instead of a full image. The encoding is
a sketch the stackable model can fill in. The smaller generative model only
needs to produce the sketch.

This is of course similar to standard transfer learning. Transfer learning was
applied to generative models for example in
[arXiv:1906.11613 [cs.LG]](https://arxiv.org/abs/1906.11613).
The stackable version could be faster if the generative model is trained to
directly to reproduce the distribution of the sketches. The full image does not
need to be produced during training.

## Does it work?

Maybe?

A stackable VAE trained on the
[Open Image Dataset](https://opensource.google/projects/open-images-dataset)
can reproduce images from the Celeb_A dataset quite well.
I trained a VAE to on the encoded representation in a
couple of hours on my laptop. There are examples of these
images in [samples/global_variation_vae.png].
Training the small model several orders of magnitude
faster than training the full model.

Naturally the full model will suffer any inaccuracy of the stackable model.
But this is quite small. Even after training the stackable model on a different
dataset, it seems do reproduce Celeb_A images really well.

This leaves two problems. If the loss function for the small generative model
only depends on the encodings, any difference in the space of the encodings
needs to translate simply into a difference in the space of the images.

Using a variational autoencoder helps here, the distances in the encoding space
have some meaning, but it does not fully solve the problem. Different features
can still have different importance to the image space, so loss in one space
does not map neatly into a loss in the other space.

In effect this means that training takes more iterations, but the result in the
limit of infinite data is the same. Since the iterations are a lot faster, the
situation is not that bad. You can also refine using the full image loss.

The other problem is that the small generative model still needs to be able to
produce the sketch. So only some part of the full generative model can be
trained using the auxiliary task of autoencoding. The rest still needs to be
learned by the small model.

So there is some limit to how small you can make the generative model. My
examples are below this limit and still take a while to run on a laptop without
a GPU.

## Image and feature variation

The implementation of the stackable model in the repository generates a mean
encoding with the shape (IMG_SIZE/2^N, IMG_SIZE/2^N, N_encode) and a standard
deviation with the shape (N_encode). The variation same variation is applied to
the features on all the pixels in the encoding.

Another simple option would be to produce a separate variance for each pixel
pixel and feature. This would correspond to the assumption that each feature at
each pixel in independent, which is not really a reasonable expectation.

The sample image [samples/local_variation_vae.png] is the result of training a
VAE on a stackable VAE with a separate variance for each pixel. The result has
interesting aberrations, but cannot really say anything more useful about this.

The current implementation instead assumes that each pixel maximally correlates
with each other pixel. This is perhaps more sensible, but might not be the best
prescription.

## Usage

To install the layered autoencoder run, clone the repository and in the
resulting directory run
```
pip install .
```

A new autoencoder is created using
```
import stacked_autoencoder.models
sae = stacked_autoencoder.models.Autoencoder(size=size, n_out=encoding_size, n_scalings = scalings)
```
This can be trained on a dataset or a list of tensors:
```
autoencoder.train(train_dataset, epochs)
validation_loss = autoencoder.evaluate(validation_dataset)
autoencoder.save()
```

You can also load an existing autoencoder using
```
sae = Autoencoder(save_path = path, load = True)
```
or just load the encoder and decoder separately using the standard API
```
encoder = tf.keras.models.load_model(path+"/encoder")
decoder = tf.keras.models.load_model(path+"/decoder")
```


## Stacked tests

To verify that training models below a stacked layer is faster, I have run a
couple of quick experiments. These are stacks of standard autoencoders, not
variation ones. The one layer model is a standard autoencoder. The
two layer model has two stacked autencoders, with the first one reducing the
image size by a factor of 8. Each layers in the three layer model reduces the
image size by a factor of 4.

There is not visible reduction in the time for training the first encoding
layer, even though the model is somewhat smaller. The subsequent layers are
significantly faster to train. The full model suffers some reduction in
validation accuracy.

| Model       | level| input shape  | scaling factor | Validation loss | Time per batch |
| -----       | -----| -----------  | -------------- | --------------- | -------------- |
| One layer   | 1    | (64, 64,  3) | flatten        |  0.0594         | 0.485          |
| Two layer   | 1    | (64, 64,  3) | 8              |  0.0178         | 0.485          |
| Two layer   | 2    | ( 8,  8, 32) | 8 (flatten)    |  0.0709         | 0.099          |
| Three layer | 1    | (64, 64,  3) | 4              |  0.000641       | 0.475          |
| Three layer | 2    | (16, 16, 32) | 4              |  0.0178         | 0.130          |
| Three layer | 3    | ( 4,  4, 32) | 4 (flatten)    |  0.0709         | 0.098          |


## BEGAN experiment

A stackable GAN can provide a base for a boundary equilibrium GAN (BEGAN), since
there is no difference between the intermediate representations of the encoder
and the decoder.

Time per batch with full BEGAN: 36
Time per batch with loss from image, encoding 8x8x64: 12
Time per batch with loss from encoding, encoding 8x8x64: 0.134
