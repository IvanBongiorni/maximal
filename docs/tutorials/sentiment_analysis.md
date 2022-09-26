# A Transformer Neural Network for Sentiment Analysis.

Author: [Ivan Bongiorni](https://github.com/IvanBongiorni) - 2022-09-25.

Open this tutorial on [[Google Colaboratory](https://colab.research.google.com/drive/1j0vDhAZX7Ni_sdCDb0C1veMtW3FEXlRD?usp=sharing).

<br>

The structure of this tutorial is loosely based on [this official Keras Notebook](https://keras.io/examples/nlp/text_classification_with_transformer/).

Let's see how to build and train a Keras model containing a `TransformerLayer` from `maximal`, using the `OriginalTransformerSchedule`.

First, I will import the main libraries I need:
```
import warnings

import numpy as np
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
```

We need TensorFlow for the model structure:
```
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, GlobalAveragePooling1D, Dropout, Dense
```

And then we can import `maximal` layers.

The central class in this tutorial is the Transformer layer. In order to include a `TransformerLayer` in a Keras model we need to import a `PositionalEmbedding` layer too. This layer will produce embeddings of words and their relative positions to inform our Attention mechanism.

Additionally, the learning rate schedule of the [original Transformer paper](https://arxiv.org/abs/1706.03762) is added for demonstration purposes.
```
import maximal as max
from maximal.layers import PositionalEmbedding, TransformerLayer
from maximal.schedules import OriginalTransformerSchedule
```

<br>

### Load IMDB dataset for Sentiment Analysis
```
vocab_size = 20000
maxlen = 200  # input length

(x_train, y_train), (x_val, y_val) = tf.keras.datasets.imdb.load_data(num_words=vocab_size)

print(len(x_train), "Training sequences")
print(len(x_val), "Validation sequences")

x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_val = tf.keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)
```

<br>

### Build the model
Let's first specify some hyperparams about Transformer's hidden size (depth), the number of Attention Heads, and the size of the internal Pointwise Feed-Forward Net:
```
model_depth = 32
num_heads = 4
ff_dim = 32
```
And then we can specify the model:
```
model = Sequential([
    Input(shape=(maxlen,)),
    PositionalEmbedding(maxlen, vocab_size, model_depth),

    TransformerLayer(model_depth, num_heads, ff_dim),

    GlobalAveragePooling1D(),
    Dropout(0.1),
    Dense(20, activation="relu"),
    Dropout(0.1),
    Dense(2, activation="softmax")
])
```
We are now ready to compile our `model`:
```
# Set learning rate schedule
transformer_schedule = OriginalTransformerSchedule(model_depth)

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=transformer_schedule),
    loss = tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)
```

<br>

### Training

Now the model is ready for training:
```
history = model.fit(
    x_train, y_train, batch_size=32, epochs=4, validation_data=(x_val, y_val)
)
```
