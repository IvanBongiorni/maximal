# maximal

See the [Official Documentation site](https://ivanbongiorni.github.io/maximal/)

Current version: **0.3.0 (Beta)**

A TensorFlow-compatible Python library that provides models and layers to implement custom Transformer neural networks.

Built on TensorFlow 2.

<br>

# Installation
Its installation is straightforward:

```
pip install maximal
```

<br>

# How to use it?
`maximal` is commonly called as:

```
import maximal as max
```

<br>

# Documentation
An official [documentation website] with explanations and tutorials is on the way.

<br>

# Elements

In `layers.py`:
- `SelfAttention`: `keras.Layer`, computes *Scaled Dot-Product Attention*.

- `MultiHeadSelfAttention`: `keras.Layer`, it is a concatenation of `SelfAttention` layers, resized back to original input shape through linear transformation.

- `PositionalEmbedding`: `keras.Layer`, implements double Embedding layers used in Transformers literature, for tokens and positions. Positional encoding is learned through a `tf.keras.layers.Embedding()` layer, instead of deterministic positional encoding in the original paper.

- `TransformerLayer`: `keras.Layer` single Transformer Encoder piece. It can be used inside any `Sequential()` model in Keras.

In `schedules.py`:
- `OriginalTransformerSchedule`: `keras.Layer` implements the learning rate schedule of the original Transformer paper. It is taken from this [official TensorFlow tutorial](https://www.tensorflow.org/text/tutorials/transformer).

<br>

# Requirements
```
numpy
tensorflow >= 2.0
```

<br>

# Author
Ivan Bongiorni. [LinkedIn](https://www.linkedin.com/in/ivan-bongiorni-b8a583164/)

<br>

# License
2020 Ivan Bongiorni

This repository is licensed under the MIT license. See [LICENCE.txt]() for further details.
