# SelfAttention()

Implements Self-Attention from *Attention is All You Need* paper.

Inherits from [tensorflow.keras.layers.Layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer).

`__init__` arguments:
  - `depth`: (int) model depth, corresponds to the size of the Embedding.

`call` arguments:
  - `x`: (np.array, tf.tensor) tensor to be attentioned

calling the Layer returns:
  - `attention`: (tf.tensor) attention output 
