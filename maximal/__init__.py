import numpy as np
import tensorflow as tf

__version__ = "1.0"


def causal_attention_mask(self, batch_size, seq_len):
    """
    Generates Mask tensor to implement causal Attention mechanism.

    Args:
        batch_size (int): size of the input batch to be attentioned
        seq_len (int): lenght of the input sequence

    Returns:
        (tf.Tensor): causal attention mask
    """
    return 1 - tf.linalg.band_part(tf.ones((batch_size, seq_len, seq_len)), -1, 0)
