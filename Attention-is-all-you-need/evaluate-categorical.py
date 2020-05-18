import tensorflow as tf
from model import Transformer
import preprocess_categorical

tokenizer_en = preprocess_categorical.get_tokenizers()
train_dataset, val_dataset = preprocess_categorical.get_data()

# Hyperparameters
d_model = 512
epochs = 100
input_vocab_size = tokenizer_en.get_vocab_size() + 2

transformer = Transformer(
            d_model = d_model,
            encoder_vocab_size = input_vocab_size,
            decoder_vocab_size = 10,
            layer_count = 6,
            head_count = 8,
            batch_size = 64,
            padding_value = 0,
            encoder_only = True
)

checkpoint_path = "./checkpoints"
checkpoint = tf.train.Checkpoint(transformer = transformer)
checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=3)
if checkpoint_manager.latest_checkpoint:
  checkpoint.restore(checkpoint_manager.latest_checkpoint)
  print("[checkpoint restored]")
test_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
test_accuracy.reset_states()


for (batch, (inp, tar)) in enumerate(val_dataset):
  predictions = transformer(inp, training=False)
  test_accuracy(tar, predictions)
  if batch % 50 == 0:
    print("Test Accuracy: ", test_accuracy.result())