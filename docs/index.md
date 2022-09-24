
Version: **0.3 (beta)**

<br>

## Structure of the library:

- `layers`
  - `SelfAttention()`. Implements *Scaled Dot-Product Attention* as in the original [Transformer paper](https://arxiv.org/abs/1706.03762), where Q, K, V are the same tensor. (`tensorflow.keras.layers.Layer`)
  - `MultiHeadSelfAttention()`. *Multi Head Self-Attention* layer, as a concatenation of `SelfAttention()` layers. (`tensorflow.keras.layers.Layer`)
  - `PositionalEmbedding()`. Double `Embedding()` layer to learn token and position representations. This differs from the original formulation of *positional encoding*, and is based on the SOTA of Transformers.
  - `TransformerLayer`. Transformer Encoder layer.
  - TransformerDecoderLayer (coming soon)
  - FNetLayer (coming soon)
  - ConformerLayer (coming soon)

- models (coming soon)
  - Transformer
  - GPT
  - FNet
  - Conformer

- `schedules`
  - `OriginalTransformerSchedule()`. Learning rate schedule used in the original [Transformer paper](https://arxiv.org/abs/1706.03762), to be fed into a Keras optimizer. (`tf.keras.optimizers.schedules.LearningRateSchedule`)


<br>

## Tutorials
- Sentiment Analysis with a Transformer Neural Network in TensorFlow. (Google Colab Notebook)
