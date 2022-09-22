"""
Layers:

- SelfAttention()
- MultiHeadSelfAttention()
- PositionalEmbedding()
- TransformerLayer()

TODO:
expand with: TransformerDecoderLayer(), FNetLayer
"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
    call args:
        x: (np.array, tf.tensor) tensor to be attentioned
    """
    def __init__(self, depth):
        super(SelfAttention, self).__init__()
        self.dense_q = tf.keras.layers.Dense(depth, activation='linear')
        self.dense_k = tf.keras.layers.Dense(depth, activation='linear')
        self.dense_v = tf.keras.layers.Dense(depth, activation='linear')

    def call(self, x):
        WQ = self.dense_q(x)
        WV = self.dense_v(x)
        WK = self.dense_k(x)

        # Scaled Dot-Product Attention
        d_k = tf.cast(tf.shape(WK)[-1], tf.float32) # cast to float32 prevents error

        attention = tf.matmul(WQ, WK, transpose_b=True)
        attention = attention / tf.math.sqrt(d_k)
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


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    """
    __init__args:
        heads: (int) number of Self-Attention heads
        depth: (int) depth of the model (corresponds to embedding size)

    call args:
        x: (np.array, tf.tensor) tensor to be attentioned

    Returns:
        attention_output: (tf.tensor) Multi-Head Attention tensor
    """
    def __init__(self, heads, depth):
        super(MultiHeadSelfAttention, self).__init__()
        self.attention_heads = [SelfAttention(depth=depth) for _ in range(heads)]
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.dense = tf.keras.layers.Dense(depth, activation='linear')

    def call(self, x):
        # Iterate call of each Self-Attention layer in self.attention_heads list
        attention_results = []
        for layer in self.attention_heads:
            attention_results.append(layer(x))

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


class TransformerLayer(tf.keras.layers.Layer):
    """
    Transformer Encoder Layer.

    __init__ args:
        depth: (int) depth of the model (corresponds to embedding size)
        num_heads: (int) number of attention heads
        pwff_nodes: (int) size of Dense ReLU layer in Pointwise FF block
        rate: (float) dropout probability. Defaults to 0.1 as in original paper

    call args:
        input_tensor: (tf.tensor) input tensor (usually from PositionalEmbedding layer)

    Returns:
        pwff_output: (tf.tensor) Layer output
    """
    def __init__(self, depth, num_heads, pwff_nodes, rate=0.1):
        super(TransformerLayer, self).__init__()
        self.attention = MultiHeadSelfAttention(heads=num_heads, depth=depth)
        self.pwff = tf.keras.models.Sequential([
            tf.keras.layers.Dense(pwff_nodes, activation="relu"),
            tf.keras.layers.Dense(depth, activation='linear')
        ])
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
        pwff_output = self.pwff(tensor_attentioned)
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
