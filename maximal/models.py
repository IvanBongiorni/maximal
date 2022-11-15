"""
Maximal models (inherited from tensorflow.keras.models.Model class) and their tools.

Elements:
- save(): function; saves a hybrid model (tf.keras and maximal layers wrapped in a tf.keras model)
- load(): function: loads a hybrid model (tf.keras and maximal layers wrapped in a tf.keras model)
"""
import os
import tensorflow as tf

import layers


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


class GPT(tf.keras.models.Model):
    def __init__(vocab_size, n_blocks, depth, heads, ff_nodes, rate=0.1, out_activation='softmax'):
        super(GPT, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.n_blocks = n_blocks
        self.depth = depth
        self.heads = heads
        self.ff_nodes = ff_nodes
        self.rate = 0.1
        self.out_activation = out_activation

        self.embedding_layer = PositionalEmbedding()
        self.gpt_layers = [TransformerLayer(depth=self.depth, heads=self.heads, ff_nodes=self.ff_nodes) for _ in range(self.n_blocks)]
        self.classification_layer = Dense(self.vocab_size, activation=self.out_activation)

        self.mask = attention_mask(mask)

    def call(self):
        x = self.embedding_layer(x)

        for layer in gpt_layers:
            x = layer(x, self.mask)

        token_prediction = self.classification_layer(x)
        return token_prediction

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'vocab_size': self.vocab_size,
            'n_blocks': self.n_blocks,
            'depth': self.depth,
            'heads': self.heads,
            'ff_nodes': self.ff_nodes,
            'rate': self.rate,
            'out_activation' = self.out_activation,

        })
        return config
