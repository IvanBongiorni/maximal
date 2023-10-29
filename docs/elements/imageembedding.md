# ImageEmbedding()

Takes a batch of images of shape `[batch_size, height, width, channels]`, breaks them into image patches, and combines their linear projections with positional ambeddings.
This Embedding layer is supposed to be the input of Vision Transformer architectures.

Inherits from [tensorflow.keras.layers.Layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer).

## Arguments

`__init__` arguments:
  - `image_shape` (Iterable[int]) shape of the image as `[height, width]`
  - `patch_size` (int) size of a (squared) image patches
  - `depth`: (int) Embedding size - more generally, model depth in original Transformer paper.
  - `padding` (str) padding type. Take two values: "SAME" (apply zero pad), or "VALID" (crop image to specified shape)

`call` arguments:
  - `inputs`: (np.array, tf.Tensor) batch of images.

## Returns
  - `image_embeddings`: (tf.tensor) Transformer Embeddings (word meaning + position).

## Used in tutorial
- [Building a Vision Transformer for Image Classification](https://ivanbongiorni.github.io/maximal/tutorials/vision_transformer.html). ([Google Colab](https://colab.research.google.com/drive/1LWFDOLJ9HGUXsHttLatU39HLlTRcTjEG?usp=sharing))
