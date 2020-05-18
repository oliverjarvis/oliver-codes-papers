
import tensorflow as tf
#from modules.multiheadattention import MultiHeadAttention
from modules.multiheadattention import MultiHeadAttention

def feed_forward(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])

class EncoderDecoder(tf.keras.layers.Layer):
    def __init__(self, 
            stack_n,
            head_count,
            d_model,
            batch_size,
            encoder_only = False,
            **kwargs
            ):

        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder_only = encoder_only
        #layers
        self.encoderStack = EncoderStack(stack_n = stack_n, head_count = head_count, d_model=d_model, batch_size = batch_size)
        if not self.encoder_only:
            self.decoderStack = DecoderStack(stack_n = stack_n, head_count = head_count, d_model = d_model, batch_size = batch_size)

    def call(self, 
            encoder_input,
            decoder_input = None,
            source_mask = None,
            target_mask = None,
            target_subsequent_mask = None 
            ):

        encoder = self.encoderStack(encoder_input, source_mask)
        output = encoder
        if not self.encoder_only:
            decoder = self.decoderStack(decoder_input, encoder, target_mask, target_subsequent_mask)
            output = decoder
        
        return output

class EncoderStack(tf.keras.layers.Layer):
    def __init__(
            self, 
            stack_n, 
            head_count,
            d_model,
            batch_size,
            **kwargs
        ):

        super(EncoderStack, self).__init__(**kwargs)

        self.encoder_layers = []
        for _ in range(stack_n):
            encoder = Encoder(head_count=head_count, d_model = d_model, batch_size = batch_size)
            self.encoder_layers.append(encoder)

    def call(self, tensor, source_mask):

        for layeridx in range(len(self.encoder_layers)):
            tensor = self.encoder_layers[layeridx](tensor, source_mask)

        return tensor

class DecoderStack(tf.keras.layers.Layer):
    def __init__(
            self, 
            stack_n, 
            head_count, 
            d_model, 
            batch_size,
            **kwargs
        ):

        super(DecoderStack, self).__init__(**kwargs)

        self.decoder_layers = []

        for _ in range(stack_n):
            decoder = Decoder(head_count=head_count, d_model = d_model, batch_size = batch_size)
            self.decoder_layers.append(decoder)

    def call(self, decoder_input, encoder_input, target_mask, target_subsequent_mask):
        
        for layeridx in range(len(self.decoder_layers)):
            decoder_input = self.decoder_layers[layeridx](decoder_input, encoder_input, target_mask, target_subsequent_mask)
        
        return decoder_input

class Encoder(tf.keras.layers.Layer):
    def __init__(
            self, 
            head_count,
            d_model,
            batch_size,
            **kwargs
        ):

        super(Encoder, self).__init__(**kwargs)

        #layers
        self.multiheadattention = MultiHeadAttention(head_count = head_count, d_model = d_model, batch_size = batch_size)
        self.layerNorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layerNorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn = feed_forward(d_model, 2048)
        self.dropout1 = tf.keras.layers.Dropout(rate=0.1)
        self.dropout2 = tf.keras.layers.Dropout(rate=0.1)
        
    def call(self, x, source_mask):

        sublayer1 = self.multiheadattention(x, x, x, source_mask)
        sublayer1 = self.dropout1(sublayer1)
        out1 = self.layerNorm1(x + sublayer1)
        sublayer2 = self.ffn(out1)
        sublayer2 = self.dropout2(sublayer2)
        out2 = self.layerNorm2(out1 + sublayer2)
        
        return out2

class Decoder(tf.keras.layers.Layer):
    def __init__(
            self, 
            head_count,
            d_model,
            batch_size, 
            **kwargs
        ):

        super(Decoder, self).__init__(**kwargs)

        #layers
        self.multiheadattention = MultiHeadAttention(head_count = head_count, d_model = d_model, batch_size = batch_size)
        self.masked_multiheadattention = MultiHeadAttention(head_count = head_count, d_model = d_model, batch_size = batch_size)
        self.layerNorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layerNorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layerNorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn = feed_forward(d_model, 2048)
        self.dropout1 = tf.keras.layers.Dropout(rate=0.1)
        self.dropout2 = tf.keras.layers.Dropout(rate=0.1)
        self.dropout3 = tf.keras.layers.Dropout(rate=0.1)

    def call(self, decoder_input, encoder_input, target_mask, target_subsequent_mask):

        sublayer1 = self.masked_multiheadattention(decoder_input, decoder_input, decoder_input, target_subsequent_mask)
        sublayer1 = self.dropout1(sublayer1)
        out1 = self.layerNorm1(sublayer1 + decoder_input)
        sublayer2 = self.multiheadattention(encoder_input, encoder_input, out1, target_mask)
        sublayer2 = self.dropout2(sublayer2)
        out2 = self.layerNorm2(sublayer2 + out1)
        sublayer3 = self.ffn(out2)
        sublayer3 = self.dropout3(sublayer3)
        out3 = self.layerNorm3(sublayer3 + out2)

        return out3