import tensorflow as tf
import numpy as np
import math

class EmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim, **kwargs):
        super(EmbeddingLayer, self).__init__(**kwargs)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_scale = self.output_dim

        #layers
        self.embedding = tf.keras.layers.Embedding(input_dim=self.input_dim, output_dim=self.output_dim)
        self.positionalEncoding = positional_encoding(input_dim, output_dim)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_scale, tf.float32))
        
        return x + self.positionalEncoding[:, :seq_len, :]

def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
  
  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  
  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
  pos_encoding = angle_rads[np.newaxis, ...]
    
  return tf.cast(pos_encoding, dtype=tf.float32)

'''
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model, **kwargs):

        super(PositionalEncoding, self).__init__(**kwargs)
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.positional_embedding = positional_embedding(vocab_size, self.d_model)
            
    def call(self, x):
        # input length is defined as the second dimension, since the first is the batch sizes
        input_length = tf.shape(x)[1]
        # retrieving only the `input_length` first positional embeddings
        x += self.positional_embedding[:, :input_length, :]
        return x

def positional_embedding(vocab_size, d_model):
    positional_encoding = np.zeros((vocab_size, d_model))
    for token in range(vocab_size):
        for index in range(d_model):
            operation = np.sin if index % 2 == 0 else np.cos
            inner = token / (10000 ** (2 * index / np.float32(d_model)))
            positional_encoding[token, index] = operation(inner)
    return tf.cast(positional_encoding[tf.newaxis, :], dtype=tf.float32)
'''

