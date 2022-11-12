"""
Maximal models (inherited from tensorflow.keras.models.Model class) and their tools.

Elements:
- save(): function; saves a hybrid model (tf.keras and maximal layers wrapped in a tf.keras model)
- load(): function: loads a hybrid model (tf.keras and maximal layers wrapped in a tf.keras model)
"""
import os
import pickle
import tensorflow as tf


def save(model, path):
    """
    Runs basic .save() method of tf.keras Model's.
    If no extension is specified it gives '.maximal' to it.
    """
    import os
    import warnings

    filename = os.path.basename(path)

    if '.' in filename:
        warnings.warn(f"No path specified. Model '{filename}' has been saved as '{filename}.maximal'")
        path += '.maximal'

    model.save(path)

    return None


def load(path):
    """ Wrapper for completion """
    import tensorflow as tf
    return model = tf.keras.models.load_model(path)
