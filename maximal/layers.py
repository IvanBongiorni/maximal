"""
Layers:

- PositionalEmbedding(): performs sum of token and position embeddings to prepare
    vectorized text for transformer layers
- Attention(): layer for Scaled Dot Product Attention
- MultiHeadAttention(): concatenations of multiple Attention() heads
- TransformerLayer(): implementation of Encoder layer
- GPTLayer(): implementation of Decoder layer

TODO:
expand with: TransformerDecoderLayer(), FNetLayer
"""
import tensorflow as tf


class PositionalEmbedding(tf.keras.layers.Layer):
    """
    __init__ args:
        maxlen: (int) maximum length of sentence
        vocab_size: (int) vocabulary size
        depth: (int) Embedding size - more generally, model depth in original paper

    call args:
        x: (np.array) input tokens

    Returns:
        embedding: (tf.tensor) Transformer Embeddings (word meaning + position)
    """
    def __init__(self, maxlen, vocab_size, depth):
        super(PositionalEmbedding, self).__init__()
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
            'token_embedding': self.token_embedding,
            'position_embedding': self.position_embedding,
        })


class Attention(tf.keras.layers.Layer):
    """
    Scaled Dot Product Attention layer (tf.keras layer)
    Applies linear transformation to input tensors and applies formula from
    "Attention is All You Need".
    In pure Self-Attention Q, K, V are the same tensor. In Deoder layers the second
    attention mechanism combines Encoder and Decoder information and they differ.

    __init__ args:
        depth: (int) depth of the model (usually corresponds to embedding size)

    call args:
        q: (np.array, tf.tensor) Query matrix
        k: (np.array, tf.tensor) Key matrix
        v: (np.array, tf.tensor) Values matrix
        mask: (np.array, tf.tensor) mask of future attention tokens - must be used
            for in Decoder layers (i.e. GPTLayer's) to prevent attention mechanism
            from peeking into the future (defaults to None)

    Returns:
        attention: (tf.tensor) attention tensor
    """
    def __init__(self, depth):
        super(Attention, self).__init__()
        self.depth = depth

        self.dense_q = tf.keras.layers.Dense(depth, activation='linear')
        self.dense_k = tf.keras.layers.Dense(depth, activation='linear')
        self.dense_v = tf.keras.layers.Dense(depth, activation='linear')

    def call(self, q, k, v, mask=None):
        WQ = self.dense_q(q)
        WV = self.dense_v(v)
        WK = self.dense_k(k)

        # Scaled Dot-Product Attention
        d_k = tf.cast(tf.shape(WK)[-1], tf.float32) # cast to float32 prevents error

        attention = tf.matmul(WQ, WK, transpose_b=True)
        attention = attention / tf.math.sqrt(d_k)

        if mask is not None:
            attention = attention + mask * -1e09

        attention = tf.nn.softmax(attention)
        attention = tf.matmul(attention, WV)
        return attention

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'dense_q': self.dense_q,
            'dense_k': self.dense_k,
            'dense_v': self.dense_v,
        })


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Implements Multi Head Attention as a concatenation of Attention() layers.

    __init__args:
        heads: (int) number of Self-Attention heads
        depth: (int) depth of the model (corresponds to embedding size)

    call args:
        q: (np.array, tf.tensor) Query matrix
        k: (np.array, tf.tensor) Key matrix
        v: (np.array, tf.tensor) Values matrix
        mask: (np.array, tf.tensor) mask of future attention tokens - must be used
            for in Decoder layers (i.e. GPTLayer's) to prevent attention mechanism
            from peeking into the future (defaults to None)

    Returns:
        attention_output: (tf.tensor) Multi-Head Attention tensor
    """
    def __init__(self, heads, depth):
        super(MultiHeadAttention, self).__init__()
        self.heads = heads
        self.depth = depth

        self.attention_heads = [Attention(depth=depth) for _ in range(heads)]
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.dense = tf.keras.layers.Dense(depth, activation='linear')

    def call(self, q, k, v, mask=None):
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
            'attention_heads': self.attention_heads,
            'concat': self.concat,
            'dense': self.dense,
        })


class TransformerLayer(tf.keras.layers.Layer):
    """
    Transformer Encoder Layer.
    It contains only one Self-Attention mechanism, unmasked. Useful for the
    implementation of BERT-like models.

    __init__ args:
        depth: (int) depth of the model (corresponds to embedding size)
        num_heads: (int) number of attention heads
        ff_nodes: (int) size of Dense ReLU layer in Pointwise FF block
        rate: (float) dropout rate (defaults to 0.1 as in original paper)

    call args:
        input_tensor: (np.array, tf.tensor) input tensor (usually from PositionalEmbedding layer)
        mask: (np.array, tf.tensor) mask matrix to mask future tokens (defaults to None)

    Returns:
        pwff_output: (tf.tensor) Layer output
    """
    def __init__(self, depth, num_heads, ff_nodes, rate=0.1):
        super(TransformerLayer, self).__init__()
        self.depth = depth
        self.num_heads = num_heads
        self.ff_nodes = ff_nodes
        self.rate = rate

        self.attention = MultiHeadAttention(heads=num_heads, depth=depth)
        self.pointwise_ffnn = tf.keras.models.Sequential([
            tf.keras.layers.Dense(ff_nodes, activation="relu"),
            tf.keras.layers.Dense(depth, activation='linear')
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, input_tensor, mask=None):
        # Self-Attention part
        multihead_attention = self.attention(input_tensor, input_tensor, input_tensor, mask)
        multihead_attention = self.dropout1(multihead_attention)
        tensor_attentioned = input_tensor + multihead_attention
        tensor_attentioned = self.layernorm1(tensor_attentioned)

        # Pointwise FFNN part
        pwff_output = self.pointwise_ffnn(tensor_attentioned)
        pwff_output = self.dropout2(pwff_output)
        pwff_output = tensor_attentioned + pwff_output
        pwff_output = self.layernorm2(pwff_output)

        return pwff_output

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'attention': self.attention,
            'pwff': self.pwff,
            'layernorm1': self.layernorm1,
            'layernorm2': self.layernorm2,
            'dropout1': self.dropout1,
            'dropout2': self.dropout2,
        })


class TransformerDecoderLayer(tf.keras.layers.Layer):

    def __init__(self, depth,  num_heads, ff_nodes, rate=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.depth = depth
        self.num_heads = num_heads
        self.ff_nodes = ff_nodes
        self.rate = rate

        self.masked_attention = MultiHeadAttention(heads=num_heads, depth=depth)
        self.cross_attention = MultiHeadAttention(heads=num_heads, depth=depth)

        self.pointwise_ffnn = tf.keras.models.Sequential([
            tf.keras.layers.Dense(ff_nodes, activation="relu"),
            tf.keras.layers.Dense(depth, activation='linear')
        ])

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, context, mask):
        # First Attention mechanism: masked Self-Attention
        masked_attention = self.masked_attention(q=x, k=x, v=x, mask=mask)
        masked_attention = self.dropout1(masked_attention)
        tensor_attentioned = x + masked_attention
        tensor_attentioned = self.layernorm1(tensor_attentioned)

        # Second Attention mechanism combining Encoder and Decoder information
        cross_attention = self.cross_attention(q=tensor_attentioned, k=context, v=context)
        cross_attention = self.dropout2(cross_attention)
        tensor_attentioned = tensor_attentioned + cross_attention
        tensor_attentioned = self.layernorm2(tensor_attentioned)

        # Final part is Pointwise FFNN
        pwff_output = self.pointwise_ffnn(tensor_attentioned)
        pwff_output = self.dropout3(pwff_output)
        pwff_output = tensor_attentioned + pwff_output
        pwff_output = self.layernorm3(pwff_output)

        return pwff_output

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'pwff': self.pwff
        })
