import tensorflow as tf
from tensorflow.keras import layers, Model, preprocessing, Sequential
import math
import numpy as np

#We might need to do glorot initialization
#missing regularization techniques

def scaled_dot_product_attention(Q, K, V, mask=False):
    attention = tf.linalg.matmul(Q, K, transpose_b=True)
    attention = attention / tf.math.sqrt(tf.cast(K.shape[1], dtype=tf.float32))

    if mask:
        row, column = np.triu_indices(attention.shape[0], k=1)
        attention[:, row, column] = -np.inf

    attention = tf.nn.softmax(attention)
    return tf.linalg.matmul(attention, V)

class MultiHeadAttention(layers.Layer):
    def __init__(self, h=8, mask=None, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.h = h
        self.mask = mask
    def call(self, V, K, Q):
        d_k = 64
        d_v = d_k
        heads = []
        for i in range(self.h):
            linear_Q = tf.keras.layers.Dense(units=d_k)(Q)
            linear_K = tf.keras.layers.Dense(units=d_k)(K)
            linear_V = tf.keras.layers.Dense(units=d_v)(V)
            head = scaled_dot_product_attention(linear_Q, linear_K, linear_V, self.mask)
            heads.append(head)
        MultiHead = tf.concat(heads,axis=-1)
        return MultiHead

class Encoder(layers.Layer):
    def __init__(self, h=8, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.h = h
    def call(self, x):
        sublayer1 = MultiHeadAttention(h=self.h)(x, x, x)
        sublayer1 = tf.add(x, sublayer1)
        sublayer1 = layers.LayerNormalization()(sublayer1)
        sublayer2 = layers.Dense(units=2048, activation=tf.nn.relu)(sublayer1)
        sublayer2 = layers.Dense(units=512)(sublayer2)
        output = tf.add(sublayer1, sublayer2)
        return output

class Decoder(layers.Layer):
    def __init__(self, h=8, mask=None, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.mask = mask
        self.h = h
    def call(self, tensors):
        x = tensors[0]
        encoder_layer = tensors[1]
        sublayer1 = MultiHeadAttention(h=self.h, mask=None)(x, x, x)
        sublayer1 = tf.add(x, sublayer1)
        sublayer1 = layers.LayerNormalization()(sublayer1)
        sublayer2 = MultiHeadAttention(h=self.h)(encoder_layer, encoder_layer, x)
        sublayer2 = tf.add(x, sublayer1)
        sublayer2 = layers.LayerNormalization()(sublayer2)
        sublayer3 = layers.Dense(units=2048, activation=tf.nn.relu)(sublayer2)
        sublayer3 = layers.Dense(units=512)(sublayer3)
        output = tf.add(sublayer2, sublayer3)
        return output

class EmbeddingLayer(layers.Layer):
    def __init__(self, input_dim, output_dim, input_length, scale, **kwargs):
        super(EmbeddingLayer, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_length = input_length
        self.scalefactor = scale
    def call(self, x):
        x = tf.keras.layers.Embedding(input_dim=self.input_dim, output_dim=self.output_dim, input_length = self.input_length)(x)
        x *= tf.math.sqrt(tf.cast(self.scalefactor, tf.float32))
        return x

class PositionalEncoding(layers.Layer):
    def __init__(self, input_length, scale, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.input_length = input_length
        self.scalefactor = scale
        self.positional_embedding = positional_embedding(self.input_length, self.scalefactor)
    def call(self, x):
        x += self.positional_embedding
        return x

def positional_embedding(input_length, scalefactor):
    positional_encoding = np.zeros((input_length, scalefactor))
    for token in range(input_length):
        for index in range(scalefactor):
            operation = math.sin if index % 2 == 0 else math.cos
            inner = token / (10000 ** (2 * index / scalefactor))
            positional_encoding[token, index] = operation(inner)
    return positional_encoding

class Transformer(tf.keras.Model):
    def __init__(self, input_shape = None, input_tensor = None, vocab_size = 10000, embedding_dimension = 512, maxlength = 512):
        super(Transformer, self).__init__()
        self.d_model = embedding_dimension
        self.maxlength = maxlength
        self.vocab_size = vocab_size
        self.Embeddings = EmbeddingLayer(input_dim=self.vocab_size, output_dim=512, input_length=self.maxlength, scale=self.d_model)
        self.PositionalEncoder = PositionalEncoding(input_length=self.maxlength, scale=self.d_model)
    def transformermodel(self):
        #My decoder and encoder functions do not work. But for the obvious reason
        #that they are destroyed as soon as the function is exited.
        #Possible change structure:
            # - start with input for both encoder and decoder
            # - then parse all the encoders to all the decoders.
                # - decode(encode())
        #input should be the whole sequence
        encoder_input = tf.keras.Input(shape=(self.maxlength,), name='encoder_input')
        embeddings = self.Embeddings(encoder_input)
        embeddings = self.PositionalEncoder(embeddings)
        #encoder = self.encoder(embeddings, h = 8, stack_n = 6)
        x = Encoder(h=8)(embeddings)
        x = Encoder(h=8)(x)
        x = Encoder(h=8)(x)
        x = Encoder(h=8)(x)
        x = Encoder(h=8)(x)
        x = Encoder(h=8)(x)
        decoder_input = tf.keras.Input(shape=(self.maxlength,), name='decoder_input')
        embeddings = self.Embeddings(decoder_input)
        embeddings = self.PositionalEncoder(embeddings)
        decoder = self.decoder(embeddings, x, h = 8, stack_n = 6, mask = False)

        outputs = tf.keras.layers.Dense(units=decoder.shape[-1], activation=tf.nn.softmax)(decoder)
        model = tf.keras.Model(inputs=[encoder_input, decoder_input], outputs=outputs, name='Transformer')
        return model

    def encoder(self, x, h=8, stack_n=6):
        #to make this work we might have to return N layers.
        for i in range(stack_n):
            x = Encoder(h=h)(x)
        return x
    
    def decoder(self, decoder_input, encoder_input, h=8, stack_n = 6, mask=False):
        for i in range(stack_n):
            x = Decoder(h=8, mask=False)([decoder_input, encoder_input])
        return x

transformer = Transformer(input_length=256)
model = transformer.transformermodel()
print(model.summary())
tf.keras.utils.plot_model(model, 'my_first_model.png', show_shapes=True, expand_nested=True)