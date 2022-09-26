# PositionalEmbedding()

Double `tensorflow.keras.layers.Embedding()` layer to learn token and position representations, respectively. This differs from the original formulation of *positional encoding*, and is based on the SOTA of Transformers.

Inherits from [tensorflow.keras.layers.Layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer).

## Arguments

`__init__` arguments:
  - `maxlen`: (int) maximum length of sentence.
  - `vocab_size`: (int) vocabulary size.
  - `depth`: (int) Embedding size - more generally, model depth in original paper.

`call` arguments:
  - `x`: (np.array) input tokens.

## Returns
  - `embedding`: (tf.tensor) Transformer Embeddings (word meaning + position).

## Used in tutorial
- A Transformer Neural Network for Sentiment Analysis. (Google Colab | NBViewr)