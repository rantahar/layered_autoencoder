
# Stackable autoencoder

The stackable autoencoder encodes an image into a smaller image with
a given number of features per pixel. It is otherwise a standard autoencoder,
it consists of an encoder and a decoder, which are trained to reproduce the
original image from the encoding.

Since the encoding has the same number of dimensions as an image, most model
architectures with image input can be stacked on top of it. That is, they can
take the encoded representation as input. So the stacked model could be a
classifier, a variational autoencoder (VAE) or a generative adversarial model
(GAN).
The stackable model only needs to be trained once and
reduces the computational effort of training the other model.

This is partly based on
[arXiv:1906.11613 [cs.LG]](https://arxiv.org/abs/1906.11613), where a GAN was
trained using the intermediate representations of an autoencoder.

## Training

Each stacked layer is a standard autencoder. It consists of an encoder `E` and
a decoder `D`. It is trained to deviation between an original image `i` and the
reproduction `D(E(i))`. 

A subsequent stacked layer would have it's own encoder `E_2` and decoder `D_2`.
It is trained to minimize the squared mean deviation between `E(i)` and
`D_2(E_2(E(i)))`.

The full validation loss of the second layer is still `(i - D(D_2(E_2(E(i)))))^2`.
While it is sufficient to train each stacked layer independently, the full
validation loss of the second layer will be larger than the validation loss
of the first layer.

## Stacked tests

To verify that training models below a stacked layer is faster, I have run a
couple of quick experiments. The one layer model is a standard autoencoder. The
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

## BEGAN

A stackable GAN can provide a base for a boundary equilibrium GAN (BEGAN), since
there is no difference between the intermediate representations of the encoder
and the decoder.
