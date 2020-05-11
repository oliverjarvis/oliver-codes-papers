import tensorflow as tf
from model import Transformer
import preprocess

tokenizer_en, tokenizer_de = preprocess.get_tokenizers()
train_dataset, val_dataset = preprocess.get_data()

# Hyperparameters
d_model = 512
epochs = 100
input_vocab_size = tokenizer_en.get_vocab_size() + 2
target_vocab_size = tokenizer_de.get_vocab_size() + 2
MAX_LENGTH = 40

transformer = Transformer(
            d_model = d_model,
            encoder_vocab_size = input_vocab_size,
            decoder_vocab_size = target_vocab_size,
            layer_count = 6,
            head_count = 8,
            batch_size = 64,
            padding_value = 0
)

checkpoint_path = "./checkpoints"
checkpoint = tf.train.Checkpoint(transformer = transformer)
checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=3)
if checkpoint_manager.latest_checkpoint:
  checkpoint.restore(checkpoint_manager.latest_checkpoint)
  print("[checkpoint restored]")

def evaluate(inp_sentence):
  start_token = [tokenizer_en.get_vocab_size()]
  end_token = [tokenizer_en.get_vocab_size() + 1]
  
  # inp sentence is portuguese, hence adding the start and end token
  inp_sentence = start_token + tokenizer_en.encode(inp_sentence).ids + end_token
  encoder_input = tf.expand_dims(inp_sentence, 0)
  
  # as the target is english, the first word to the transformer should be the
  # english start token.
  decoder_input = [tokenizer_de.get_vocab_size()]
  output = tf.expand_dims(decoder_input, 0)
    
  for i in range(MAX_LENGTH):
  
    # predictions.shape == (batch_size, seq_len, vocab_size)
    predictions = transformer(encoder_input, output, 
                                 True)
    
    # select the last word from the seq_len dimension
    predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
    
    # return the result if the predicted_id is equal to the end token
    if predicted_id == tokenizer_de.get_vocab_size()+1:
      return tf.squeeze(output, axis=0)
    
    # concatentate the predicted_id to the output which is given to the decoder
    # as its input.
    output = tf.concat([output, predicted_id], axis=-1)

  return tf.squeeze(output, axis=0)

def translate(sentence, plot=''):
  result = evaluate(sentence)
  
  predicted_sentence = tokenizer_de.decode([i for i in result 
                                            if i < tokenizer_de.get_vocab_size()])  

  print('Input: {}'.format(sentence))
  print('Predicted translation: {}'.format(predicted_sentence))

translate("hello how are you doing? I am okay.")