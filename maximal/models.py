"""
This submodule is currently used for functions to load TensorFlow models that contain Maximal layers.

Future releases will contain full Maximal models inherited from tf.keras.models.Model class
"""
import json
import warnings

import h5py
import numpy as np
import tensorflow as tf
import maximal
from maximal import layers


def load(path: str):
    """
    Automatically loads a model in TensorFlow's .5 format that contains Maximal layers.
    The used doesn't need to specify which ones, the function reads the .h5 file,
    lists all custom layers, and dynamically imports the right classes to make sure
    loading works.

    Args: path (str): path to .5 model file
    Returns: tf.keras.models.Model (optionally with Maximal layers).
    """
    # Extract model architecture/configuration
    with h5py.File(path, 'r') as f:
        model_config = f.attrs['model_config'] #.decode("utf-8")

    config = json.loads(model_config)

    # Extract list of unique custom layers
    model_layers = config['config']['layers']
    model_layers = [layer['class_name'] for layer in model_layers]
    model_layers = list(filter(lambda layer: layer.startswith("Custom"), model_layers))
    model_layers = [layer.replace("Custom>", "") for layer in model_layers]
    model_layers = list(set(model_layers))

    if len(model_layers) > 0:
        # Produce list of custom objects to be loaded in tf model
        custom_objects = {}
        for layer in model_layers:
            custom_objects[layer] = getattr(maximal.layers, layer)

        # Model with custom objects can now be loaded without issues
        model = tf.keras.models.load_model(path, custom_objects=custom_objects)
        return model
    else:
        try:
            model = tf.keras.models.load_model(path)
            warnings.warn(f"maximal load(): model loaded from {path} doesn't contain any Maximal object.")
            return model
        except Exception as ex:
            print(f"Maximal: Loading model failed. Exception: {ex}")

