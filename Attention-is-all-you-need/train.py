from model import Transformer
import tensorflow as tf
import numpy as np

vocab_size = 100

#tf.executing_eagerly()

model = Transformer(
            d_model = 500,
            encoder_vocab_size = vocab_size,
            decoder_vocab_size = vocab_size,
            layer_count = 6,
            head_count = 8,
            batch_size = 5,
            padding_value = 0
)

def data_gen(V, nsim):
    data = []
    for _ in range(nsim):
        random = np.random.randint(1, V, size=(20)).tolist()
        data.append(random)
    return (data, data)

x_train, y_train = data_gen(100, 200)

a = [[1,2,3],[2,3,4]]
#train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
#train_dataset = train_dataset.batch(5)
#print(list(train_dataset.as_numpy_iterator()))

model.compile(optimizer=tf.keras.optimizers.Adam(),  
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit([x_train, y_train], steps_per_epoch=10, epochs=2)

print("hello")