
# Official Documentation.
Version: **1.0**

A TensorFlow-compatible Python library that provides models and layers to implement custom Transformer neural networks. Built on [TensorFlow 2](https://www.tensorflow.org/api_docs/python/tf).

## Structure of the library:

- `layers`
  - class [SelfAttention](https://ivanbongiorni.github.io/maximal/elements/selfattention.html). Self-Attention implementation.
  - class [MultiHeadSelfAttention](https://ivanbongiorni.github.io/maximal/elements/multiheadselfattention.html). Multi Head Self-Attention implementation.
  - class [PositionalEmbedding](https://ivanbongiorni.github.io/maximal/elements/positionalembedding.html). Embeddings of tokens end positions.
  - class [TransformerLayer](https://ivanbongiorni.github.io/maximal/elements/transformerlayer.html). Transformer Encoder layer.
  - class [GPTLayer]((https://ivanbongiorni.github.io/maximal/elements/gptlayer.html)). GPT Layer.

- `schedules`
  - class [OriginalTransformerSchedule](https://ivanbongiorni.github.io/maximal/elements/originaltransformerschedule.html). From the original [Transformer paper](https://arxiv.org/abs/1706.03762).

- `models` (coming soon)

## Tutorials
- [A Transformer Neural Network for Sentiment Analysis](https://ivanbongiorni.github.io/maximal/tutorials/sentiment_analysis.html). ([Google Colab](https://colab.research.google.com/drive/1j0vDhAZX7Ni_sdCDb0C1veMtW3FEXlRD?usp=sharing))
- [Neural Text Generation with a Custom GPT](https://ivanbongiorni.github.io/maximal/tutorials/gpt.html). ([Google Colab](https://drive.google.com/file/d/1GOrseFbhD01E3LGR69y6EFDEx9ZDuP-d/view?usp=sharing))
- [Save and load models](https://ivanbongiorni.github.io/maximal/tutorials/save_and_load.html).

<br>

### Author
**Ivan Bongiorni**
<br>Data Scientist, Associate Director
<br>UBS, Emerging Solutions
<br>Zurich, Switzerland.

[GitHub](https://github.com/IvanBongiorni)
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
[LinkedIn](https://www.linkedin.com/in/ivan-bongiorni-b8a583164/)

ivanbongiorni@protonmail.com
