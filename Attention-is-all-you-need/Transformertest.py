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
    def __init__(self, h=8, mask=None, batch_size = 0, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.h = h
        self.mask = mask
        self.batch_size = batch_size
    
    def split_heads(self, tensor, batch_size, depth):
        #reshape into the desired dimension
        #we want to preserve the batches, and split the embedding dimension
        #into (heads, depth)
        tensor = tf.reshape(tensor, (batch_size, -1, self.h, depth))
        return tf.transpose(tensor, perm=[0,2,1,3])

    def call(self, V, K, Q):
        d_k = 64
        d_v = d_k
        heads = []
        dimension = K.shape[-1]
        depth = K.shape[-1] // self.h

        Q = tf.keras.layers.Dense(units=d_k)(Q)
        K = tf.keras.layers.Dense(units=d_k)(K)
        V = tf.keras.layers.Dense(units=d_v)(V)

        Q = self.split_heads(Q, self.batch_size, depth)
        K = self.split_heads(K, self.batch_size, depth)
        V = self.split_heads(V, self.batch_size, depth)

        attention = scaled_dot_product_attention(Q, K, V, self.mask)

        attention = tf.transpose(attention, perm=[0,2,1,3])
        attention = tf.reshape(attention, (self.batch_size, -1, dimension))
        
        return attention

class EncoderDecoder(layers.Layer):
    def __init__(self, n_stack = 6, h = 8, mask = False, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.Encoder = EncoderStack(stack_n = 6, h = 8)
        self.Decoder = DecoderStack(stack_n = 6, h=8, mask=False)
    def call(self, embeddings):
        encoder_embeddings = embeddings[0]
        decoder_embeddings = embeddings[1]
        encoder = self.Encoder(encoder_embeddings)
        decoder = self.Decoder([decoder_embeddings, encoder])
        return decoder

class Encoder(layers.Layer):
    def __init__(self, h = 8, **kwargs):
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
    def __init__(self, h=8, mask=False, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.mask = mask
        self.h = h
    def call(self, tensors):
        x = tensors[0]
        encoder_layer = tensors[1]
        sublayer1 = MultiHeadAttention(h=self.h, mask=self.mask)(x, x, x)
        sublayer1 = tf.add(x, sublayer1)
        sublayer1 = layers.LayerNormalization()(sublayer1)
        sublayer2 = MultiHeadAttention(h=self.h)(encoder_layer, encoder_layer, x)
        sublayer2 = tf.add(x, sublayer1)
        sublayer2 = layers.LayerNormalization()(sublayer2)
        sublayer3 = layers.Dense(units=2048, activation=tf.nn.relu)(sublayer2)
        sublayer3 = layers.Dense(units=512)(sublayer3)
        output = tf.add(sublayer2, sublayer3)
        return output

class EncoderStack(layers.Layer):
    def __init__(self, stack_n = 6, h = 8, **kwargs):
        super(EncoderStack, self).__init__(**kwargs)
        self.encoder_layers = []
        for i in range(stack_n):
            encoder = Encoder(h=8)
            self.encoder_layers.append(encoder)
    def call(self, tensor):
        for layeridx in range(len(self.encoder_layers)):
            tensor = self.encoder_layers[layeridx](tensor)
        return tensor

class DecoderStack(layers.Layer):
    def __init__(self, stack_n = 6, h = 8, mask=False, **kwargs):
        super(DecoderStack, self).__init__(**kwargs)
        self.decoder_layers = []
        for i in range(stack_n):
            decoder = Decoder(h=8, mask=mask)
            self.decoder_layers.append(decoder)
    def call(self, tensor):
        decoding = tensor[0]
        encoding = tensor[1]
        for layeridx in range(len(self.decoder_layers)):
            decoding = self.decoder_layers[layeridx]([decoding, encoding])
        return decoding

class EmbeddingLayer(layers.Layer):
    def __init__(self, input_dim, output_dim, input_length, scale, **kwargs):
        super(EmbeddingLayer, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_length = input_length
        self.scalefactor = scale
        self.embedding = tf.keras.layers.Embedding(input_dim=self.input_dim, output_dim=self.output_dim, input_length = self.input_length)
    def call(self, x):
        return self.embedding(x) * tf.math.sqrt(tf.cast(self.scalefactor, tf.float32))

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
    def __init__(self, input_shape = None, input_tensor = None, vocab_size = 10000, embedding_dimension = 512, input_length = 512):
        super(Transformer, self).__init__()
        self.d_model = embedding_dimension
        self.maxlength = input_length
        self.vocab_size = vocab_size

        #Layer initialization
        self.Embeddings = EmbeddingLayer(input_dim=self.vocab_size, output_dim=512, input_length=self.maxlength, scale=self.d_model)
        self.PositionalEncoder = PositionalEncoding(input_length=self.maxlength, scale=self.d_model)
        self.EncoderDecoder = EncoderDecoder(n_stack = 6, h = 8, mask = False)
        self.outputdense = tf.keras.layers.Dense(self.vocab_size)
    def call(self, inputs):
        encoder_input = inputs[0]
        decoder_input = inputs[1]
        #Input layers
        #MISTAKE. SHOULD NOT USE tf.keras.input
        #this is for the functional API, which we are not using
        #encoder_input = tf.keras.Input(shape=(self.maxlength,), name='encoder_input')
        #decoder_input = tf.keras.Input(shape=(self.maxlength,), name='decoder_input')
        print(encoder_input.shape)
        encoder_embeddings = self.Embeddings(encoder_input)
        encoder_embeddings = self.PositionalEncoder(encoder_embeddings)
        decoder_embeddings = self.Embeddings(decoder_input)
        decoder_embeddings = self.PositionalEncoder(decoder_embeddings)
        
        #encoderdecoder = self.EncoderDecoder([encoder_embeddings, decoder_embeddings])

        outputs = self.outputdense(encoderdecoder)
        return outputs

transformer = Transformer(input_length=256)
transformer.build(input_shape=[(None, 256), (None, 256)])
print(transformer.summary())
tf.keras.utils.plot_model(transformer, to_file = "out.png", expand_nested = True)