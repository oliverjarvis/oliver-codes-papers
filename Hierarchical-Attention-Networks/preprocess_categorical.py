import os
import pickle
import numpy as np
from pathlib import Path
import tensorflow as tf
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
import glob
import re
from collections import Counter
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
#go through all words
#keep words with more than five occurences
#idx to word
#word to idx

class WordTokenizer:
    def __init__(self, threshold, unk_token=None):
        self.threshold = threshold
        self.unk_token = unk_token
        self.tokens = Counter()
        self.index_to_token = dict()
        self.token_to_index = dict()
    def initFromDataFiles(self, data):
        for d in data:
            with open(d) as f:
                for line in f.readlines():
                    self.tokens.update(text_to_word_sequence(line))
        self.tokens = [k for k, v in dict(self.tokens).items() if v >= self.threshold]
        if self.unk_token:
            self.tokens.append(self.unk_token)
        for index, word in enumerate(self.tokens):
            self.index_to_token[index] = word
            self.token_to_index[word] = index
        return self
    def encode(self, sentence):
        tokenized = text_to_word_sequence(sentence)
        encoded = []
        for token in tokenized:
            if self.unk_token:
                encoded.append(self.token_to_index.get(token, len(self.token_to_index) - 1))
            else:
                if token in self.token_to_index.keys():
                    encoded.append(self.token_to_index[token])
        return encoded

    def decode(self, ids):
        return " ".join([self.index_to_token.get(id) for id in ids])
    
    def vocab_size(self):
        return len(self.tokens)

english_dir = "en"

train_pos = "./data/train/pos"
train_neg = "./data/train/neg"
test_pos = "./data/test/pos"
test_neg = "./data/test/neg"

train_dir = "./data/train"
train_dir_pos = glob.glob(train_pos + "/*.txt", recursive=True)
train_dir_neg = glob.glob(train_neg + "/*.txt", recursive=True)
train_files = glob.glob(train_dir + "/*/*.txt", recursive=True)

test_dir = "./data/test"
test_dir_pos = glob.glob(test_pos + "/*.txt", recursive=True)
test_dir_neg = glob.glob(test_neg + "/*.txt", recursive=True)
test_files = glob.glob(test_dir + "/*/*.txt", recursive=True)

MAX_DOC_LENGTH = 18
MAX_SENT_LENGTH = 150

tokenizer = WordTokenizer(5, unk_token="<unk>").initFromDataFiles(train_files)

if  (os.path.exists("Hierarchical-Attention-Networks/train_data.pickle") and  
        os.path.exists("Hierarchical-Attention-Networks/val_data.pickle")):
    train_inputs = pickle.load(open("Hierarchical-Attention-Networks/train_data.pickle", "rb"))    
    val_inputs = pickle.load(open("Hierarchical-Attention-Networks/val_data.pickle", "rb"))
else:

    train_inputs = []
    for f in train_files:
        with open(f) as fi:
            text = fi.read()
            lines = re.split(r"\.+|!+|\?+", text)
            document = [tokenizer.encode(l) for l in lines if len(l) > 0]
            if len(document) > MAX_DOC_LENGTH:
                document = document[:MAX_DOC_LENGTH]
            document = tf.keras.preprocessing.sequence.pad_sequences(document, maxlen=MAX_SENT_LENGTH, padding="post")
            pad_size = MAX_DOC_LENGTH - len(document)
            document = np.pad(document, ((0,pad_size), (0, 0)))
            train_inputs.append(document)

    val_inputs = []
    for f in test_files:
        with open(f) as fi:
            text = fi.read()
            lines = re.split(r"\.+|!+|\?+", text)
            document = [tokenizer.encode(l) for l in lines if len(l) > 0]
            if len(document) > MAX_DOC_LENGTH:
                document = document[:MAX_DOC_LENGTH]
            document = tf.keras.preprocessing.sequence.pad_sequences(document, maxlen=MAX_SENT_LENGTH, padding="post")
            pad_size = MAX_DOC_LENGTH - len(document)
            document = np.pad(document, ((0,pad_size), (0, 0)))
            val_inputs.append(document)

    pickle.dump(train_inputs, open("Hierarchical-Attention-Networks/train_data.pickle", "wb+"))    
    pickle.dump(val_inputs, open("Hierarchical-Attention-Networks/val_data.pickle", "wb+"))


train_inputs = tf.data.Dataset.from_tensor_slices(train_inputs)

train_labels = tf.data.Dataset.from_tensor_slices([0 if "neg" in x else 1 for x in train_files])
train_dataset = tf.data.Dataset.zip((train_inputs, train_labels))

val_inputs = tf.data.Dataset.from_tensor_slices(val_inputs)
val_labels = tf.data.Dataset.from_tensor_slices([0 if "neg" in x else 1 for x in test_files])
val_dataset = tf.data.Dataset.zip((val_inputs, val_labels))

train_examples, val_examples = train_dataset, val_dataset

BUFFER_SIZE = 20000
BATCH_SIZE = 64

train_preprocessed = (
    train_examples
    .shuffle(BUFFER_SIZE))

val_preprocessed = (
    train_examples
    .shuffle(BUFFER_SIZE))

train_dataset = (train_preprocessed
                .batch(BATCH_SIZE)
                .prefetch(tf.data.experimental.AUTOTUNE))

val_dataset = (val_preprocessed
                .take(100)
                .batch(BATCH_SIZE))
test_dataset = (val_preprocessed
                .skip(100)
                .take(100)
                .batch(BATCH_SIZE))

def get_data():
    return train_dataset, val_dataset, test_dataset

def get_tokenizer():
    return tokenizer

import argparse
