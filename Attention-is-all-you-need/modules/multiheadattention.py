import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

def scaled_dot_product_attention(Q, K, V, mask):
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    attention = tf.linalg.matmul(Q, K, transpose_b=True)
    attention = attention / tf.math.sqrt(dk)
    if mask is not None:
        
        attention += mask * -np.inf

    attention = tf.nn.softmax(attention, axis=-1)
    return tf.linalg.matmul(attention, V)

class MultiHeadAttention(layers.Layer):
    def __init__(self, head_count, d_model, batch_size, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.head_count = head_count
        self.d_model = d_model
        self.batch_size = batch_size
        self.QDense = tf.keras.layers.Dense(units=self.d_model)
        self.KDense = tf.keras.layers.Dense(units=self.d_model)
        self.VDense = tf.keras.layers.Dense(units=self.d_model)
    
    def split_heads(self, tensor, batch_size, depth):
        # Reshape into the desired dimension
        # We want to preserve the batches, and split the embedding dimension
        # Into (heads, depth)
        tensor = tf.reshape(tensor, (batch_size, -1, self.head_count, depth))
        return tf.transpose(tensor, perm=[0,2,1,3])

    def call(self, V, K, Q, mask):
        depth = tf.cast(self.d_model // self.head_count, tf.int32)
        self.batch_size = tf.shape(Q)[0]
        Q = self.QDense(Q)
        K = self.KDense(K)
        V = self.QDense(V)

        Q = self.split_heads(Q, self.batch_size, depth)
        K = self.split_heads(K, self.batch_size, depth)
        V = self.split_heads(V, self.batch_size, depth)

        attention = scaled_dot_product_attention(Q, K, V, mask)

        attention = tf.transpose(attention, perm=[0,2,1,3])
        attention = tf.reshape(attention, (self.batch_size, -1, self.d_model))
        
        return attention