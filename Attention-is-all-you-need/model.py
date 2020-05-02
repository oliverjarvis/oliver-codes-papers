"""
Note in regards to weight tying:
    - It's not the default because it requires that source and target vocabularies have the same size which is not always true. source: guillaumekln
    - Shared Embeddings: When using BPE with shared vocabulary we can share the same weight vectors between the source / target / generator. 
      See the (cite) for details. ()
"""
import tensorflow as tf
import numpy as np
from modules.embeddings import EmbeddingLayer
from modules.encodedecode import EncoderDecoder

def create_masks(source, target, padding_value):
    source_mask = tf.cast(tf.math.equal(source, 0), tf.float32)
    target_mask = tf.cast(tf.math.equal(target, 0), tf.float32)
    length = tf.shape(target)[-1]
    print(length.shape)
    target_subsequent_mask = tf.linalg.band_part(tf.ones((length, length)), -1, 0)
    target_subsequent_mask = tf.maximum(target_subsequent_mask, target_mask)
    return (source_mask, target_mask, target_subsequent_mask)    

class Transformer(tf.keras.models.Model):
    def __init__(
            self, 
            d_model,
            encoder_vocab_size,
            decoder_vocab_size,
            layer_count,
            head_count,
            batch_size,
            padding_value
        ):

        super(Transformer, self).__init__(name="Transformer")

        #Model parameters
        self.d_model = d_model
        self.layer_count = layer_count
        self.encoder_vocab_size = encoder_vocab_size
        self.decoder_vocab_size = decoder_vocab_size
        self.head_count = head_count
        self.padding_value = padding_value

        #Layer definitions
        self.encoderEmbedding = EmbeddingLayer(input_dim=self.encoder_vocab_size, output_dim=self.d_model)
        self.decoderEmbedding = EmbeddingLayer(input_dim=self.decoder_vocab_size, output_dim=self.d_model)
        self.encoderDecoder = EncoderDecoder(stack_n = layer_count, head_count = head_count, d_model=d_model, batch_size = batch_size)
        self.fullyConnected = tf.keras.layers.Dense(self.decoder_vocab_size)

    def call(self, data_inputs, training):
        ## expected input_shape (None, None)
        encoder_input = data_inputs[0]
        decoder_input = data_inputs[1]
        source_mask, target_mask, target_subsequent_mask = create_masks(encoder_input, decoder_input, self.padding_value)

        encoder_embeddings = self.encoderEmbedding(encoder_input)
        decoder_embeddings = self.decoderEmbedding(decoder_input)
        
        encoderdecoder = self.encoderDecoder(
            encoder_embeddings,
            decoder_embeddings,
            source_mask, 
            target_mask, 
            target_subsequent_mask
            )

        last_layer = self.fullyConnected(encoderdecoder)

        return last_layer