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
        self.positionalEncoding = PositionalEncoding(input_dim, output_dim)

    def call(self, x):
        x = self.embedding(x)
        x = self.positionalEncoding(x)
        return x * tf.math.sqrt(tf.cast(self.d_scale, tf.float32))

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
            operation = math.sin if index % 2 == 0 else math.cos
            inner = token / (10000 ** (2 * index / d_model))
            positional_encoding[token, index] = operation(inner)
    return positional_encoding