import tensorflow as tf
import os
import pickle
import numpy as np
from tensorflow.keras.layers import GRU, Bidirectional, TimeDistributed, AdditiveAttention, Embedding, Dense
import preprocess_categorical
from gensim.models import KeyedVectors
#Word encoder -> need word2vec embeddings
#Linear transformation
#Bidirectional GRU
#Badhanau
#Turn into sentence vector (look at dimension output from badhanau, if not good we need to do a weighted sum)
#Bidirectional GRU on sentence vectors
#Badhanau

tokenizer = preprocess_categorical.get_tokenizer()
train_dataset, val_dataset, test_dataset = preprocess_categorical.get_data()
emb_file = "Hierarchical-Attention-Networks/GoogleNews-vectors-negative300.bin" 

def embedding_matrix(token_to_index, embeddingDIM):
    word_vectors = KeyedVectors.load_word2vec_format(emb_file, binary=True)
    #using random to give the unk token some weight
    emb_matrix = np.zeros((len(token_to_index), embeddingDIM))

    for k, v in token_to_index.items():
        if v == len(token_to_index) - 1:
            continue
        if k in word_vectors:
            vector = word_vectors[k]
            emb_matrix[v,:] = vector
        else:
            print(k)

    return emb_matrix
    

class CustomAttention(tf.keras.layers.Layer):
    def __init__(self, attention_dim):
        super(CustomAttention, self).__init__()
        self.attention_dim = 100
        self.uitDense = Dense(attention_dim, activation=tf.nn.tanh)
        self.ubDense = Dense(1, use_bias=False)

    def call(self, x, training=True):
        uit = self.uitDense(x)
        wcv = self.ubDense(uit)
        wcv = tf.squeeze(wcv, axis=-1)
        wcv = tf.exp(wcv)
        ait = tf.divide(wcv, tf.reduce_sum(wcv, axis=1, keepdims=True)) #basically a softmax 
        ait = tf.expand_dims(ait,-1)
        si = tf.reduce_sum(x * ait, axis=1)
        return si
    #must be implemented because input dimensions do not match output dimensions
    #Possibly also due to some funkiness with the TimeDistributed function
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

EMBEDDING_DIM = 300

def create_model(embedding_matrix):
    sinput = tf.keras.Input(shape=(150,), dtype='int32')
    embedding_layer = Embedding(tokenizer.vocab_size(),
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            trainable=False)(sinput)
    ##Apply against sentence
    WordEncoder = Bidirectional(GRU(units=50, return_sequences=True))(embedding_layer)
    WordEncoder = CustomAttention(100)(WordEncoder)
    sEncoder = tf.keras.models.Model(inputs=sinput, outputs=WordEncoder)
    
    finput = tf.keras.Input(shape=(18, 150), dtype='int32')
    fEncoder = TimeDistributed(sEncoder)(finput)
    fEncoder = Bidirectional(GRU(units=50, return_sequences=True))(fEncoder)
    fEncoder = CustomAttention(100)(fEncoder)
    output = Dense(1, activation=tf.nn.sigmoid)(fEncoder)
    model = tf.keras.Model(inputs=finput, outputs=output)

    return model

#Implement learnong rate scheduler
#prepare text

if os.path.exists("Hierarchical-Attention-Networks/embeddings.pickle"):
    emb = pickle.load(open("Hierarchical-Attention-Networks/embeddings.pickle", "rb"))
else:
    emb = embedding_matrix(tokenizer.token_to_index, EMBEDDING_DIM)
    pickle.dump(emb, open("Hierarchical-Attention-Networks/embeddings.pickle", "wb+"))
model = create_model(emb)
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.003), loss="binary_crossentropy", metrics=['acc'])
model.summary()
history = model.fit(train_dataset, validation_data = val_dataset, epochs=200)
print("hello")

