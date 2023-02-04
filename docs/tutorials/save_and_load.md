# Saving and Loading Models.

Author: [Ivan Bongiorni](https://github.com/IvanBongiorni) - 2023-02-04.

<br>

Loading and saving maximal models (usually models that are composed of a mix of maximal and TensorFlow layers) is straightforward.
Currently, the syntax is the one from `tensorflow.keras`. After you basic imports:

```
import tensorflow as tf
import maximal
```

in order to save a `model` simply run:

```
model.save("./path/to/model.h5")
```

The `.h5` file extension is optional.

For loading the same model:

```
model = tf.keras.models.load_model('path/to/model.h5')
```

Future releases of maximal will implement saving and loading functions and a `.maximal` model format.
