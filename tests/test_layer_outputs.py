"""
Testing is based on pytest framework.
Any pull request that doesn't pass all tests must be rejected a priori.

This script tests for output shapes and data types (tf.float32)
"""
import pytest

import numpy as np
import tensorflow as tf
from maximal.layers import (
    PositionalEmbedding, ImageEmbedding,
    Attention, MultiHeadAttention,
    TransformerLayer,
    GPTLayer
)


@pytest.mark.parametrize("maxlen, vocab_size, depth", [(10, 50, 64), (15, 100, 32)])
def test_output_positionalembedding(maxlen, vocab_size, depth):
    batch_size = 5
    sequence_length = 7

    # Create a random input tensor
    x = np.random.randint(0, vocab_size, (batch_size, sequence_length))

    # Call the layer
    layer = PositionalEmbedding(maxlen=maxlen, vocab_size=vocab_size, depth=depth)
    output = layer(x)

    # Check the output shape and dtype
    assert output.shape == (batch_size, sequence_length, depth)
    assert output.dtype == tf.float32


def test_output_imageembedding(maxlen, vocab_size, depth):
    # Create a random input tensor
    x = np.random.randint(0, 100, (32, 28, 28, 3))

    layer = ImageEmbedding(image_shape=[28, 28], patch_size=4, depth=32, padding="SAME")
    output = layer(x)

    # Check the output shape and dtype
    assert output.shape == (batch_size, sequence_length, depth)
    assert output.dtype == tf.float32


def test_output_attention():
    depth = 64
    batch_size = 5
    sequence_length = 7

    # Create random input tensors
    q = tf.random.uniform((batch_size, sequence_length, depth), dtype=tf.float32)
    k = tf.random.uniform((batch_size, sequence_length, depth), dtype=tf.float32)
    v = tf.random.uniform((batch_size, sequence_length, depth), dtype=tf.float32)

    # Call the layer
    layer = Attention(depth=depth)
    output = layer(q, k, v)

    # Check the output shape and dtype
    assert output.shape == (batch_size, sequence_length, depth)
    assert output.dtype == tf.float32


def test_output_multiheadattention():
    heads = 8
    depth = 64
    batch_size = 5
    sequence_length = 7

    # Create random input tensors
    q = tf.random.uniform((batch_size, sequence_length, depth), dtype=tf.float32)
    k = tf.random.uniform((batch_size, sequence_length, depth), dtype=tf.float32)
    v = tf.random.uniform((batch_size, sequence_length, depth), dtype=tf.float32)

    # Call the layer
    layer = MultiHeadAttention(heads=heads, depth=depth)
    output = layer(q, k, v)

    # Check the output shape and dtype
    assert output.shape == (batch_size, sequence_length, depth)
    assert output.dtype == tf.float32


def test_output_transformerlayer():
    depth = 64
    heads = 8
    ff_nodes = 256
    rate = 0.1
    batch_size = 5
    sequence_length = 7

    # Create a random input tensor
    input_tensor = tf.random.uniform((batch_size, sequence_length, depth), dtype=tf.float32)

    # Call the layer
    layer = TransformerLayer(depth=depth, heads=heads, ff_nodes=ff_nodes, rate=rate)
    output = layer(input_tensor)

    # Check the output shape and dtype
    assert output.shape == (batch_size, sequence_length, depth)
    assert output.dtype == tf.float32


def test_output_gptlayer():
    depth = 64
    heads = 8
    ff_nodes = 256
    rate = 0.1
    batch_size = 5
    sequence_length = 7

    # Create a random input tensor
    input_tensor = tf.random.uniform((batch_size, sequence_length, depth), dtype=tf.float32)

    # Call the layer
    layer = GPTLayer(depth=depth, heads=heads, ff_nodes=ff_nodes, rate=rate)
    output = layer(input_tensor)

    # Check the output shape and dtype
    assert output.shape == (batch_size, sequence_length, depth)
    assert output.dtype == tf.float32
