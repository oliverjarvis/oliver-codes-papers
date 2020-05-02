import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

def scaled_dot_product_attention(Q, K, V, mask):
    attention = tf.linalg.matmul(Q, K, transpose_b=True)
    attention = attention / tf.math.sqrt(tf.cast(K.shape[1], dtype=tf.float32))
    if mask:
        row, column = np.triu_indices(attention.shape[0], k=1)
        attention[:, row, column] = -np.inf

    attention = tf.nn.softmax(attention)
    return tf.linalg.matmul(attention, V)

class MultiHeadAttention(layers.Layer):
    def __init__(self, head_count, d_model, batch_size, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.head_count = head_count,
        self.d_model = d_model,
        self.batch_size = batch_size
    
    def split_heads(self, tensor, batch_size, depth):
        # Reshape into the desired dimension
        # We want to preserve the batches, and split the embedding dimension
        # Into (heads, depth)
        tensor = tf.reshape(tensor, (batch_size, -1, self.head_count, depth))
        return tf.transpose(tensor, perm=[0,2,1,3])

    def call(self, V, K, Q, mask):
        depth = self.d_model // self.head_count

        Q = tf.keras.layers.Dense(units=self.d_model)(Q)
        K = tf.keras.layers.Dense(units=self.d_model)(K)
        V = tf.keras.layers.Dense(units=self.d_model)(V)

        Q = self.split_heads(Q, self.batch_size, depth)
        K = self.split_heads(K, self.batch_size, depth)
        V = self.split_heads(V, self.batch_size, depth)

        attention = scaled_dot_product_attention(Q, K, V, mask)

        attention = tf.transpose(attention, perm=[0,2,1,3])
        attention = tf.reshape(attention, (self.batch_size, -1, self.d_model))
        
        return attention