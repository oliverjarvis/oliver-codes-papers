import os
import numpy as np
from pathlib import Path
import tensorflow as tf
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
import glob

def createBPETokenizer(data, vocab_size, name=""):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    #Create trainer with vocav size
    trainer = trainers.BpeTrainer(vocab_size=int(vocab_size))

    #train with data
    tokenizer.train(trainer,
    	data
    )

    #save to name folder
    tokenizer.model.save(".", name)
    return tokenizer

def loadBPETokenizer(name):
    # Load a BPE Model
    vocab = name + "-vocab.json"
    merges = name + "-merges.txt"
    bpe = models.BPE(vocab, merges)

    # Initialize a tokenizer
    tokenizer = Tokenizer(bpe)

    # Customize pre-tokenization and decoding
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    return tokenizer

def encode(lang1):
    tokens = tokenizer_en.encode(str(lang1.numpy().decode("utf-8"))).ids
    tokens = tokens if len(tokens) <= 100 else tokens[:100]
    lang1 = [tokenizer_en.get_vocab_size()] + tokens + [tokenizer_en.get_vocab_size()+1]
    lang1 = np.array(lang1, dtype=np.int64)
    return lang1

def tf_encode(en, label):
    result_en = tf.py_function(encode, [en], tf.int64)
    result_en.set_shape(tf.TensorShape([None]))
    return result_en, label

english_dir = "en"

train_pos = "Attention-is-all-you-need/data/train/pos"
train_neg = "Attention-is-all-you-need/data/train/neg"
test_pos = "Attention-is-all-you-need/data/test/pos"
test_neg = "Attention-is-all-you-need/data/test/neg"

train_dir = "Attention-is-all-you-need/data/train"
train_dir_pos = glob.glob(train_pos + "/*.txt", recursive=True)
train_dir_neg = glob.glob(train_neg + "/*.txt", recursive=True)
train_files = glob.glob(train_dir + "/*/*.txt", recursive=True)

test_dir = "Attention-is-all-you-need/data/test"
test_dir_pos = glob.glob(test_pos + "/*.txt", recursive=True)
test_dir_neg = glob.glob(test_neg + "/*.txt", recursive=True)
test_files = glob.glob(test_dir + "/*/*.txt", recursive=True)

vocab_en = english_dir + "-vocab.json"
merges_en = english_dir + "-merges.txt"

if Path(vocab_en).exists and Path(merges_en).exists():
    # Load tokenizer
    tokenizer_en = loadBPETokenizer(english_dir)
else:
    # create a tokenizer
    tokenizer_en = createBPETokenizer(train_files, vocab_size=9e3, name=english_dir)


train_inputs = tf.data.TextLineDataset(train_files)
train_labels = tf.data.Dataset.from_tensor_slices([0 if "neg" in x else 1 for x in train_files])
train_dataset = tf.data.Dataset.zip((train_inputs, train_labels))

val_inputs = tf.data.TextLineDataset(test_files)
val_labels = tf.data.Dataset.from_tensor_slices([0 if "neg" in x else 1 for x in test_files])
val_dataset = tf.data.Dataset.zip((val_inputs, val_labels))

train_examples, val_examples = train_dataset, val_dataset
print(tokenizer_en.decode([171]))
BUFFER_SIZE = 20000
BATCH_SIZE = 64

train_preprocessed = (
    train_examples
    .map(tf_encode)
    .cache()
    .shuffle(BUFFER_SIZE))

val_preprocessed = (
    val_examples
    .map(tf_encode))

train_dataset = (train_preprocessed
                 .padded_batch(BATCH_SIZE, padded_shapes=([None],()))
                 .prefetch(tf.data.experimental.AUTOTUNE))


val_dataset = (val_preprocessed
               .padded_batch(BATCH_SIZE,  padded_shapes=([None], ())))


def get_data():
    return train_dataset, val_dataset

def get_tokenizers():
    return tokenizer_en