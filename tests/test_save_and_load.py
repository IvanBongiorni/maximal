"""
Testing is based on pytest framework.

Any pull request that doesn't pass all tests must be rejected a priori.
"""
import numpy as np
import tensorflow as tf
import maximal
from maximal import layers


# def test_save_and_load():
#     """
#     Creates a dummy model (name with unique timestamp), saves it and loads it.
#     """
#     import tensorflow as tf
#     from datetime import datetime
#     from tensorflow.keras.models import Sequential
#     from tensorflow.keras.layers import Input, GlobalAveragePooling1D, Dropout, Dense
#     from maximal.layers import PositionalEmbedding, TransformerLayer
#     from maximal.models import load
#
#     # Generate unique model name for this test
#     timestamp = str(datetime.now())
#     model_name = f"dummy_test_save_load_{timestamp}.h5"
#
#     # create a small dummy model and save it
#     maxlen = 32
#     vocab_size = 64
#     model_depth = 32
#     num_heads = 2
#     ff_dim = 64
#
#     dummy_model = Sequential([
#         Input(shape=(maxlen,)),
#         PositionalEmbedding(maxlen, vocab_size, model_depth),
#         TransformerLayer(model_depth, num_heads, ff_dim),
#         GlobalAveragePooling1D(),
#         Dropout(0.1),
#         Dense(20, activation="relu"),
#         Dropout(0.1),
#         Dense(2, activation="softmax")
#     ])
#     dummy_model.save(os.path.join(os.getcwd(), "test_data", model_name))
#
#     # 1st test: check saving went well
#     filenames = os.listdir(os.path.join(os.getcwd(), "test_data"))
#     assert model_name in filenames
#
#     # 2nd test: check load is well
#     dummy_model = load(os.path.join(os.getcwd(), "test_data", model_name))
#     assert isinstance(dummy_model, tf.keras.models.Model)

