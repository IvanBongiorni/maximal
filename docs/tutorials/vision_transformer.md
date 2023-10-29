# Building a Vision Transformer for Image Classification

Author: [Ivan Bongiorni](https://github.com/IvanBongiorni) - 2022-09-25.

Open this tutorial on [Google Colaboratory](https://colab.research.google.com/drive/1LWFDOLJ9HGUXsHttLatU39HLlTRcTjEG?usp=sharing).

Let's see how to build and train a Maximal-TensorFlow model containing a `ImageEmbedding` and `TransformerLayer` layers from `maximal`.

<br>

##### **DISCLAIMER: This is a tutorial to show how to implement a ViT model using Maximal layers. We are *not* trying to beat the SOTA, or to reach good model performance.**

```commandline
import math
from typing import Union, Iterable
import time

import numpy as np
import tensorflow as tf

# Model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout

# Image processing/augmentation
from tensorflow.keras.layers import Normalization, Resizing, RandomFlip, RandomRotation, RandomZoom

import matplotlib.pyplot as plt
```

```commandline
from maximal.layers import ImageEmbedding, TransformerLayer
```

## Import CIFAR10 Dataset

This is popular [dataset](https://www.cs.toronto.edu/~kriz/cifar.html) is perfect to show how easy it is to build ViT architectures in `maximal`.

It consists of 60000 colored images of `[32, 32]` size. Images are divided in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

Classes are: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

```commandline
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

input_shape = x_train.shape[1:]
num_classes = len(np.unique(y_train))

print("\nInput shape:", input_shape)
print("No. classes:", num_classes, "\n")

print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")
```

## Set Hyperparameters

```commandline
LEARNING_RATE = 0.0001

BATCH_SIZE = 64
N_EPOCHS = 4

IMAGE_SIZE = 72
PATCH_SIZE = 6

DEPTH = 256
N_HEADS = 4
FF_NODES = 1024
N_TRANSFORMER_LAYERS = 6
DROPOUT = 0.15
```

I will implement data augmentation. To save time, this step was copied from [this Keras tutorial](https://keras.io/examples/vision/image_classification_with_vision_transformer/).

```commandline
data_augmentation = Sequential(
    [
        Normalization(),
        Resizing(IMAGE_SIZE, IMAGE_SIZE),
        RandomFlip("horizontal"),
        RandomRotation(factor=0.02),
        RandomZoom(
            height_factor=0.2, width_factor=0.2
        ),
    ],
    name="data_augmentation",
)

# Compute the mean and the variance of the training data for normalization.
data_augmentation.layers[0].adapt(x_train)
```

## Model Architecture

A Neural Network is a computational graph. I will now create all of its elements:

```commandline
# Define nodes of computational graph
input_images = Input(input_shape)  # input layer

# Image Embedding to produce positional embeddings of image patches
image_embedding = ImageEmbedding([IMAGE_SIZE, IMAGE_SIZE], PATCH_SIZE, DEPTH)

# A sequence of Transformer Encoder layers
transformer_layers = []
for _ in range(N_TRANSFORMER_LAYERS):
    transformer_layers.append(TransformerLayer(DEPTH, N_HEADS, FF_NODES))

# A final Dense block to produce a classification
flatten = Flatten()

dense_block = Sequential([
    Dense(FF_NODES, activation=tf.nn.gelu),
    Dropout(DROPOUT),
    Dense(FF_NODES, activation=tf.nn.gelu),
    Dropout(DROPOUT),
], name='dense_block')

classification_layer = Dense(num_classes)
```

Now that all the elements of the computational graph are created, I will connect them together to finally build the ViT:

```commandline
# Connect nodes
augmented_batch = data_augmentation(input_images)

representation = image_embedding(augmented_batch)

for layer in transformer_layers:
    representation = layer(representation)

representation = flatten(representation)

representation = dense_block(representation)

classification = classification_layer(representation)
```

```commandline
vision_transformer = Model(
    inputs=input_images,
    outputs=classification
)
```

## Training

Since `maximal.layers` are also `tensorflow.keras.layers`, we can train our ViT as any common Keras model.

```commandline
vision_transformer.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")
)

history = vision_transformer.fit(
    x=x_train,
    y=y_train,
    batch_size=BATCH_SIZE,
    epochs=N_EPOCHS,
    validation_split=0.1
)
```



















