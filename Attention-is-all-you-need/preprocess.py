import os
from pathlib import Path
import tensorflow as tf
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

MAX_LENGTH = 40

def createBPETokenizer(data, vocab_size, name=""):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    #Create trainer with vocav size
    trainer = trainers.BpeTrainer(vocab_size=int(vocab_size))

    #train with data
    tokenizer.train(trainer, [
    	data
    ])

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

def encode(lang1, lang2):
    lang1 = [tokenizer_en.get_vocab_size()] + tokenizer_en.encode(str(lang1.numpy())).ids + [tokenizer_en.get_vocab_size()+1]

    lang2 = [tokenizer_de.get_vocab_size()] + tokenizer_de.encode(str(lang2.numpy())).ids + [tokenizer_de.get_vocab_size()+1]

    return lang1, lang2

def tf_encode(en, de):
    result_en, result_de = tf.py_function(encode, [en, de], [tf.int64, tf.int64])
    result_en.set_shape([None])
    result_de.set_shape([None])

    return result_en, result_de

def filter_max_length(x, y, max_length=MAX_LENGTH):
    return tf.logical_and(tf.size(x) <= max_length,
                            tf.size(y) <= max_length)

english_dir = "en"
german_dir = "de"

eval_en = "Attention-is-all-you-need/data/eval.en"
eval_de = "Attention-is-all-you-need/data/eval.de"
test_en = "Attention-is-all-you-need/data/test.en"
test_de = "Attention-is-all-you-need/data/test.de"

vocab_en = english_dir + "-vocab.json"
merges_en = english_dir + "-merges.txt"
vocab_de = german_dir + "-vocab.json"
merges_de = german_dir + "-merges.txt"

if Path(vocab_en).exists and Path(merges_en).exists() and Path(vocab_de).exists() and Path(merges_de).exists():
    # Load tokenizer
    tokenizer_en = loadBPETokenizer(english_dir)
    tokenizer_de = loadBPETokenizer(german_dir)
else:
    # create a tokenizer
    tokenizer_en = createBPETokenizer(eval_en, vocab_size=3e5, name=english_dir)
    tokenizer_de = createBPETokenizer(eval_de, vocab_size=3e4, name=german_dir)


train_inputs = tf.data.TextLineDataset("Attention-is-all-you-need/data/eval.en")
train_outputs = tf.data.TextLineDataset("Attention-is-all-you-need/data/eval.de")
train_dataset = tf.data.Dataset.zip((train_inputs, train_outputs))

val_inputs = tf.data.TextLineDataset("Attention-is-all-you-need/data/test.en")
val_outputs = tf.data.TextLineDataset("Attention-is-all-you-need/data/test.de")
val_dataset = tf.data.Dataset.zip((val_inputs, val_outputs))

train_examples, val_examples = train_dataset, val_dataset

BUFFER_SIZE = 20000
BATCH_SIZE = 64

train_preprocessed = (
    train_examples
    .map(tf_encode) 
    .filter(filter_max_length)
    # cache the dataset to memory to get a speedup while reading from it.
    .cache()
    .shuffle(BUFFER_SIZE))

val_preprocessed = (
    val_examples
    .map(tf_encode)
    .filter(filter_max_length))      

train_dataset = (train_preprocessed
                 .padded_batch(BATCH_SIZE, padded_shapes=([None], [None]))
                 .prefetch(tf.data.experimental.AUTOTUNE))


val_dataset = (val_preprocessed
               .padded_batch(BATCH_SIZE,  padded_shapes=([None], [None])))

def get_data():
    return train_dataset, val_dataset

def get_tokenizers():
    return tokenizer_en, tokenizer_de