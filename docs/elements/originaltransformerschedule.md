# OriginalTransformerSchedule

Learning rate schedule used in the original [Transformer paper](https://arxiv.org/abs/1706.03762). It is meant to be fed into a Keras optimizer instead of a fixed value.

Inherits from [tensorflow.keras.optimizers.schedules.LearningRateSchedule](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/LearningRateSchedule).

## Arguments
`__init__` args:
  - `depth`:(int) model depth, corresponds to the size of the Embedding.
  - `warmup_steps`: (int) Length (in steps) of warm up phase. Defaults to 4000 (as original).

## Returns
  - learning rate for each training step.

## Used in tutorial
- [A Transformer Neural Network for Sentiment Analysis](https://ivanbongiorni.github.io/maximal/tutorials/sentiment_analysis.html). ([Google Colab](https://colab.research.google.com/drive/1j0vDhAZX7Ni_sdCDb0C1veMtW3FEXlRD?usp=sharing))
