"""
Layers:

- PositionalEmbedding(): performs sum of token and position embeddings to prepare
    vectorized text for transformer layers
- Attention(): layer for Scaled Dot Product Attention
- MultiHeadAttention(): concatenations of multiple Attention() heads
- TransformerLayer(): implementation of Encoder layer
- GPTLayer(): narrower version of TransformerLayer() class with causal masking to build GPT

TODO:
expand with: TransformerDecoderLayer(), FNetLayer
"""
import numpy as np
import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class PositionalEmbedding(tf.keras.layers.Layer):
    """
    __init__ args:
        maxlen (int): maximum length of sentence
        vocab_size (int): vocabulary size
        depth (int): Embedding size - more generally, model depth in original paper

    call args:
        x (np.array): input tokens

    Returns:
        embedding (tf.tensor): Transformer Embeddings (word meaning + position)
    """
    def __init__(self, maxlen, vocab_size, depth, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.depth = depth

        self.token_embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=depth)
        self.position_embedding = tf.keras.layers.Embedding(input_dim=maxlen, output_dim=depth)

    def call(self, x):
        positions = tf.range(start=0, limit=tf.shape(x)[-1], delta=1)
        learned_positions = self.position_embedding(positions)
        embedding = self.token_embedding(x)
        embedding = embedding + learned_positions
        return embedding

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'maxlen': self.maxlen,
            'vocab_size': self.vocab_size,
            'depth': self.depth
        })
        return config


@tf.keras.utils.register_keras_serializable()
class Attention(tf.keras.layers.Layer):
    """
    Scaled Dot Product Attention layer (tf.keras layer)
    Applies linear transformation to input tensors and applies formula from
    "Attention is All You Need".
    In pure Self-Attention Q, K, V are the same tensor. In TransformerDecoderLayer's
    the second Attention mechanism is Cross-Attention, i.e. K and V are the output
    of the Encoder block.

    __init__ args:
        depth (int): depth of the model (usually corresponds to embedding size)
        use_bias (bool): whether Q,K,V linear transformations should use a bias parameter
            or not. Defaults to False following the literature

    call args:
        q (np.array, tf.tensor): corresponds to Query matrix in Scaled Dot Product Attention.
            If Key and Value are not specified then q=k=v (implementing Self-Attention).
        k (np.array, tf.tensor, None): Key matrix, defaults to None
        v (np.array, tf.tensor): Values matrix, defaults to None
        mask (np.array, tf.tensor): mask of future attention tokens - must be used
            for in Decoder layers (i.e. GPTLayer's) to prevent attention mechanism
            from peeking into the future (defaults to None)

    Returns:
        attention (tf.tensor): attention tensor

    #TODO: in maximal 2.0 change call() args to q and just another arg=None. q,k,v are 
	redundant and kept just for backward compatibility with 1.0 versions.
	in future there should be a single q input for self-attention, and a second optional
	for cross-attention.
    """
    def __init__(self, depth, use_bias=False, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.depth = depth
        self.use_bias = use_bias

        self.dense_q = tf.keras.layers.Dense(depth, activation='linear', use_bias=use_bias)
        self.dense_k = tf.keras.layers.Dense(depth, activation='linear', use_bias=use_bias)
        self.dense_v = tf.keras.layers.Dense(depth, activation='linear', use_bias=use_bias)

    def call(self, q, k=None, v=None, mask=None):
        """
        If k,v=None implements Self-Attention - i.e. use Dense(q) for all three.
        in Cross-Attention instead we don't need Dense(), since the Encoder output is
        already in the right shape
        """
        WQ = self.dense_q(q)                        # [batch_size, sequence_length, depth ]
        WK = self.dense_v(q) if k is None else k    # [batch_size, sequence_length, depth ]
        WV = self.dense_v(q) if v is None else v    # [batch_size, sequence_length, depth ]

        # Formula for Scaled Dot-Product Attention
        d_k = tf.cast(tf.shape(WK)[-1], tf.float32) # cast to float32 prevents error

        attention = tf.matmul(WQ, WK, transpose_b=True)  # [batch_size, sequence_length, sequence_length ]
        attention = attention / tf.math.sqrt(d_k)        # [batch_size, sequence_length, sequence_length ]

        if mask is not None:
            attention = tf.where(mask==1, -1e9, attention)

        attention = tf.nn.softmax(attention, axis=-1)
        attention = tf.matmul(attention, WV)  # [batch_size, sequence_length, depth ]
        return attention

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'depth': self.depth,
            'use_bias': self.use_bias
        })
        return config


@tf.keras.utils.register_keras_serializable()
class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Implements Multi Head Attention as a concatenation of Attention() layers.

    __init__args:
        heads (int): number of Self-Attention heads
        depth (int): depth of the model (corresponds to embedding size)
        use_bias (bool): whether Q,K,V linear transformations should use a bias parameter
            or not. Defaults to False following the literature

    call args:
        q (np.array, tf.tensor): Query matrix
        k (np.array, tf.tensor): Key matrix
        v (np.array, tf.tensor): Values matrix
        mask (np.array, tf.tensor): mask of future attention tokens - must be used
            for in Decoder layers (i.e. GPTLayer's) to prevent attention mechanism
            from peeking into the future (defaults to None)

    Returns:
        attention_output (tf.tensor): Multi-Head Attention tensor

    #TODO: for future maximal 2.0 see args modifications proposed in Attention()
    """
    def __init__(self, heads, depth, use_bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.heads = heads
        self.depth = depth
        self.use_bias = use_bias

        self.attention_heads = [Attention(depth=depth, use_bias=use_bias) for _ in range(heads)]
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.dense = tf.keras.layers.Dense(depth, activation='linear')

    def call(self, q, k=None, v=None, mask=None):
        # Iterate call of each Self-Attention layer in self.attention_heads list
        attention_results = []
        for layer in self.attention_heads:
            attention_results.append(layer(q, k, v, mask))

        # Results are concat plus linear transformation
        attention_output = self.concat(attention_results)
        attention_output = self.dense(attention_output)
        return attention_output

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'heads': self.heads,
            'depth': self.depth,
            'use_bias': self.use_bias
        })
        return config


@tf.keras.utils.register_keras_serializable()
class TransformerLayer(tf.keras.layers.Layer):
    """
    Transformer Encoder Layer.
    It contains only one Self-Attention mechanism, unmasked. Useful for the
    implementation of non-autoregressive models such as BERT.

    __init__ args:
        depth: (int) depth of the model (corresponds to embedding size)
        heads: (int) number of attention heads
        ff_nodes: (int) size of Dense ReLU layer in Pointwise FF block
        use_bias (bool): whether Q,K,V linear transformations should use a bias parameter
            or not. Defaults to False following the literature
        rate: (float) dropout probability. Defaults to 0.1 as in original paper

    call args:
        input_tensor (np.array, tf.tensor): input tensor (usually from PositionalEmbedding layer)

    Returns:
        pwff_output (tf.tensor): Layer output
    """
    def __init__(self, depth, heads, ff_nodes, use_bias=False, rate=0.1, **kwargs):
        super(TransformerLayer, self).__init__(**kwargs)
        self.depth = depth
        self.heads = heads
        self.ff_nodes = ff_nodes
        self.use_bias = use_bias
        self.rate = rate

        self.attention = MultiHeadAttention(heads=heads, depth=depth)

        self.pwff1 = tf.keras.layers.Dense(ff_nodes, activation="relu")
        self.pwff2 = tf.keras.layers.Dense(depth, activation='linear')

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, input_tensor):
        # Self-Attention part
        multihead_attention = self.attention(input_tensor)
        multihead_attention = self.dropout1(multihead_attention)
        tensor_attentioned = input_tensor + multihead_attention
        tensor_attentioned = self.layernorm1(tensor_attentioned)

        # Pointwise FFNN part
        pwff_output = self.pwff1(tensor_attentioned)
        pwff_output = self.pwff2(pwff_output)
        pwff_output = self.dropout2(pwff_output)
        pwff_output = tensor_attentioned + pwff_output
        pwff_output = self.layernorm2(pwff_output)

        return pwff_output

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'depth': self.depth,
            'heads': self.heads,
            'ff_nodes': self.ff_nodes,
            'use_bias': self.use_bias,
            'rate': self.rate
        })
        return config


class TransformerDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, depth, heads, ff_nodes, use_bias=False, rate=0.1, **kwargs):
        """
        Layer to compose Transformer Decoder blocks.
        Since GPTLayer() is an independent class, this Layer should be employed only for
        building complete Transformer architectures.

        __init__ args:
            depth: (int) depth of the model (corresponds to embedding size)
            heads: (int) number of attention heads
            ff_nodes: (int) size of Dense ReLU layer in Pointwise FF block
            use_bias (bool): whether Q,K,V linear transformations should use a bias parameter
                or not. Defaults to False following the literature
            rate: (float) dropout probability. Defaults to 0.1 as in original paper

        call args:
            input_tensor (tf.Tensor): input of the layer on the Decoder side (either from another
                TransformerDecoderLayer or from a PositionalEmbedding layer)
            encoder_output (tf.Tensor): output of the Transformer Encoder block, required for
                the Cross-Attention mechanism.

        Returns:
            pwff_output (tf.tensor): Layer output
        """
        super(TransformerDecoderLayer, self).__init__(**kwargs)
        self.depth = depth
        self.heads = heads
        self.ff_nodes = ff_nodes
        self.rate = rate

        self.masked_attention = MultiHeadAttention(heads=heads, depth=depth, use_bias=use_bias)
        self.cross_attention = MultiHeadAttention(heads=heads, depth=depth, use_bias=use_bias)

        self.pwff1 = tf.keras.layers.Dense(ff_nodes, activation="relu")
        self.pwff2 = tf.keras.layers.Dense(depth, activation='linear')

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def causal_attention_mask(self, batch_size, seq_len):
        return 1 - tf.linalg.band_part(tf.ones((batch_size, seq_len, seq_len)), -1, 0)

    def call(self, input_tensor, encoder_output):

        # Generate causal mask from input shape
        mask = self.causal_attention_mask(tf.shape(input_tensor)[0], tf.shape(input_tensor)[1])

        # Masked Self-Attention part
        masked_attention = self.masked_attention(
            q = input_tensor,
            k = None,
            v = None,
            mask = mask
        )
        masked_attention = self.dropout1(masked_attention)
        tensor_attentioned = input_tensor + masked_attention
        tensor_attentioned = self.layernorm1(tensor_attentioned)

        # Cross-Attention part
        cross_attention = self.cross_attention(
            q = input_tensor,
            k = encoder_output,
            v = encoder_output,
            mask = None
        )
        cross_attention = self.dropout2(cross_attention)
        tensor_attentioned = tensor_attentioned + cross_attention
        tensor_attentioned = self.layernorm2(tensor_attentioned)

        # Pointwise FFNN part
        pwff_output = self.pwff1(tensor_attentioned)
        pwff_output = self.pwff2(pwff_output)
        pwff_output = self.dropout3(pwff_output)
        pwff_output = tensor_attentioned + pwff_output
        pwff_output = self.layernorm3(pwff_output)

        return pwff_output


@tf.keras.utils.register_keras_serializable()
class GPTLayer(tf.keras.layers.Layer):
    """
    Narrower version of TransformerLayer() class with implicit application of
    causal Attention mechanism. To implement GPT models.

    __init__ args:
        depth (int): depth of the model (corresponds to embedding size)
        heads (int): number of attention heads
        ff_nodes (int): size of Dense ReLU layer in Pointwise FF block
        rate (float): dropout rate (defaults to 0.1)

    call args:
        input_tensor (np.array, tf.tensor): input tensor (usually from PositionalEmbedding layer)

    Returns:
        pwff_output: (tf.tensor) Layer output
    """
    def __init__(self, depth, heads, ff_nodes, rate=0.1, **kwargs):
        super(GPTLayer, self).__init__(**kwargs)
        self.depth = depth
        self.heads = heads
        self.ff_nodes = ff_nodes
        self.rate = rate

        self.attention = MultiHeadAttention(heads=heads, depth=depth)

        self.pwff1 = tf.keras.layers.Dense(ff_nodes, activation="relu")
        self.pwff2 = tf.keras.layers.Dense(depth, activation='linear')

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def causal_attention_mask(self, batch_size, seq_len):
        return 1 - tf.linalg.band_part(tf.ones((batch_size, seq_len, seq_len)), -1, 0)

    def call(self, input_tensor):

        # Generate causal mask from input shape
        mask = self.causal_attention_mask(tf.shape(input_tensor)[0], tf.shape(input_tensor)[1])

        # Self-Attention part
        multihead_attention = self.attention(input_tensor, input_tensor, input_tensor, mask)
        multihead_attention = self.dropout1(multihead_attention)
        tensor_attentioned = input_tensor + multihead_attention
        tensor_attentioned = self.layernorm1(tensor_attentioned)

        # Pointwise FFNN part
        pwff_output = self.pwff1(tensor_attentioned)
        pwff_output = self.pwff2(pwff_output)
        pwff_output = self.dropout2(pwff_output)
        pwff_output = tensor_attentioned + pwff_output
        pwff_output = self.layernorm2(pwff_output)

        return pwff_output

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'depth': self.depth,
            'heads': self.heads,
            'ff_nodes': self.ff_nodes,
            'rate': self.rate
        })
        return config
