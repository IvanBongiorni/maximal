# SelfAttention()

Implements *Scaled Dot-Product Attention* as in the original [Transformer paper](https://arxiv.org/abs/1706.03762), where Q, K, V are the same tensor.

Inherits from [tensorflow.keras.layers.Layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer).

## Arguments

`__init__` arguments:
  - `depth`: (int) model depth, corresponds to the size of the Embedding.

`call` arguments:
  - `x`: (np.array, tf.tensor) tensor to be attentioned.

## Returns
  - `attention`: (tf.tensor) attention output.
