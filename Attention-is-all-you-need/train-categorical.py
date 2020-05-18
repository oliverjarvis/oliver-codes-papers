from model import Transformer
import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tensorflow as tf
from utils import CustomSchedule
import preprocess_categorical

# train dataset
tokenizer_en = preprocess_categorical.get_tokenizers()
train_dataset, val_dataset = preprocess_categorical.get_data()

# Hyperparameters
d_model = 512
epochs = 100
input_vocab_size = tokenizer_en.get_vocab_size() + 2

# The learning rate is based on a learning rate scheduler
learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
# loss
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
# metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

def loss_function(real, pred):
  #mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  #mask = tf.cast(mask, dtype=loss_.dtype)
  #loss_ *= mask
  
  return loss_

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
checkpoint = tf.train.Checkpoint(transformer = transformer, optimizer = optimizer)
checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=3)
if checkpoint_manager.latest_checkpoint:
  checkpoint.restore(checkpoint_manager.latest_checkpoint)
  print("[checkpoint restored]")

def train_step(inp, tar):
  # teaching forcing
  
  with tf.GradientTape() as tape:
    predictions = transformer(inp, training=True)
    loss = loss_function(tar, predictions)
  
  gradients = tape.gradient(loss, transformer.trainable_variables)
  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
  
  train_loss(loss)
  #something fucky with the accuracy metric
  train_accuracy(tar, predictions)

for epoch in range(epochs):
  start = time.time()
  
  train_loss.reset_states()
  train_accuracy.reset_states()
  
  # inp -> portuguese, tar -> english
  for (batch, (inp, tar)) in enumerate(train_dataset):

    train_step(inp, tar)
    
    if batch % 50 == 0:
      print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, batch, train_loss.result(), train_accuracy.result()))
      
  if (epoch + 1) % 5 == 0:
    ckpt_save_path = checkpoint_manager.save()
    print ('Saving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))
    
  print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, train_loss.result(), train_accuracy.result()))
  print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))