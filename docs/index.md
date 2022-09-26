
# Official Documentation.
Version: **0.3 (beta)**

A TensorFlow-compatible Python library that provides models and layers to implement custom Transformer neural networks. Built on [TensorFlow 2](https://www.tensorflow.org/api_docs/python/tf).

## Structure of the library:

- `layers`
  - `class SelfAttention`. Self-Attention implementation.
  - `class MultiHeadSelfAttention`. Multi Head Self-Attention implementation.
  - `class PositionalEmbedding`. Embeddings of tokens end positions.
  - `class TransformerLayer`. Transformer Encoder layer.
  - class TransformerDecoderLayer (coming soon)
  - class FNetLayer (coming soon)
  - class ConformerLayer (coming soon)

- models (coming soon)
  - class Transformer
  - class GPT
  - class FNet
  - class Switch Transformer
  - class Conformer

- `schedules`
  - `class OriginalTransformerSchedule`. From the original [Transformer paper](https://arxiv.org/abs/1706.03762).

## Tutorials
- [A Transformer Neural Network for Sentiment Analysis](https://ivanbongiorni.github.io/maximal/docs/tutorials/sentiment_analysis.html). ([Google Colab](https://colab.research.google.com/drive/1j0vDhAZX7Ni_sdCDb0C1veMtW3FEXlRD?usp=sharing))

<br>

### Author
**Ivan Bongiorni**. Data Scientist, Associate Director at UBS, Zurich, Switzerland.

[GitHub](https://github.com/IvanBongiorni)
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
[LinkedIn](https://www.linkedin.com/in/ivan-bongiorni-b8a583164/)
