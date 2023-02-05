# GPTLayer

GPT Layer.

Inherits from [tensorflow.keras.layers.Layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer).

## Arguments
`__init__` args:
  - `depth` (int): depth of the model (corresponds to embedding size).
  - `heads` (int): number of attention heads.
  - `ff_nodes` (int): size of Dense ReLU layer in Pointwise FF block.
  - `rate` (float): dropout probability. Defaults to 0.1 as in original paper.

`call` args:
  - `input_tensor` (tf.tensor): input tensor (usually from PositionalEmbedding layer).

## Returns
  - (tf.tensor): Layer's output.

## Used in tutorial
- [Neural Text Generation with a Custom GPT](https://ivanbongiorni.github.io/maximal/tutorials/gpt.html). ([Google Colab](https://colab.research.google.com/drive/1pUqoGVLbSZurfcH_z1LEM5tEoXUtyzaF?usp=sharing))
