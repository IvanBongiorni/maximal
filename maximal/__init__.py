import numpy as np
import tensorflow as tf

__version__ == "0.4"


def generate_attention_mask(seq_len):
    """
    Taken from a previous version of this official TensorFlow tutorial:
        https://www.tensorflow.org/text/tutorials/transformer
    The function disappeared in latest versions

    Args:
        seq_len: (int) length of the sequence (will be size of attention tensor)

    Returns:
        attention tensor (tf.Tensor)
    """
    return 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
