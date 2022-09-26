# TransformerLayer

Transformer Encoder layer.

Inherits from [tensorflow.keras.layers.Layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer).

## Arguments
`__init__` args:
  - `depth`: (int) depth of the model (corresponds to embedding size).
  - `num_heads`: (int) number of attention heads.
  - `pwff_nodes`: (int) size of Dense ReLU layer in Pointwise FF block.
  - `rate`: (float) dropout probability. Defaults to 0.1 as in original paper.

`call` args:
  - `input_tensor`: (tf.tensor) input tensor (usually from PositionalEmbedding layer).

## Returns
  - `pwff_output`: (tf.tensor) Layer's output.
