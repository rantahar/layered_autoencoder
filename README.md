
# Stackable autoencoder

A stackable autoencoder transforms an image into a smaller encoded
representation. The 


| Model       | block | scaling factor | Validation loss | Time per batch |
| -----       | ----- | -------------- | --------------- | -------------- |
| One layer   | 1     | flatten        |  0.0594         | 0.485          |
| Two layer   | 1     | 8              |  0.0178         | 0.485          |
| Two layer   | 2     | 8 (flatten)    |  0.0709         | 0.099          |
| Three layer | 1     | 4              |  0.000641       | 0.475          |
| Three layer | 1     | 4              |  0.0178         | 0.130          |
| Three layer | 2     | 4 (flatten)    |  0.0709         | 0.098          |
