# MultiHeadSelfAttention()

*Multi Head Self-Attention* layer, it is a concatenation of `SelfAttention()` layers. (`tensorflow.keras.layers.Layer`)

Inherits from [tensorflow.keras.layers.Layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer).

## Arguments

`__init__` arguments:
  - `depth`: (int) model depth, corresponds to the size of the Embedding.

`call` arguments:
  - `x`: (np.array, tf.tensor) tensor to be attentioned.

## Returns
  - `attention`: (tf.tensor) attention output.
