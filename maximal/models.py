"""
This submodule is currently used for functions to load TensorFlow models that contain Maximal layers.

Future releases will contain full Maximal models inherited from tf.keras.models.Model class
"""
import json
import h5py
import numpy as np
import tensorflow as tf
import maximal
from maximal import layers


def load(path: str):
    """
    """

    # Extract model architecture/configuration
    with h5py.File(path, 'r') as f:
        model_config = f.attrs['model_config'] #.decode("utf-8")

    config = json.loads(model_config)

    # Extract list of unique custom layers
    layers = config['config']['layers']
    layers = [layer['class_name'] for layer in layers]
    layers = list(filter(lambda layer: layer.startswith("Custom"), layers))
    layers = [layer.replace("Custom>", "") for layer in layers]
    layers = list(set(layers))

    # Produce list of custom objects to be loaded in tf model
    custom_objects = {}
    for layer in layers:
        custom_objects[layer] = getattr(maximal.layers, layer)

    print(custom_objects)

    # Model with custom objects can now be loaded without issues
    model = tf.keras.models.load_model(path, custom_objects=custom_objects)

    return model
