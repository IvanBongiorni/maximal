import numpy as np
import tensorflow as tf

__version__ = "1.2.1"

from maximal import layers
from maximal import schedules
from maximal import models

from maximal.layers import *
from maximal.schedules import *
from maximal.models import *


def causal_attention_mask(batch_size: int, seq_len: int):
    """
    Generates Mask tensor to implement causal Attention mechanism.

    Args:
        batch_size (int): size of the input batch to be attentioned
        seq_len (int): lenght of the input sequence

    Returns:
        (tf.Tensor): causal attention mask
    """
    return 1 - tf.linalg.band_part(tf.ones((batch_size, seq_len, seq_len)), -1, 0)
