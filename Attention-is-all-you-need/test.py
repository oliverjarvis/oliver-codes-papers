import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

class MyModel(tf.keras.Model):

  def __init__(self, num_classes=10):
    super(MyModel, self).__init__(name='my_model')
    self.num_classes = num_classes
    self.dense_1 = layers.Dense(32, activation='relu')
    self.dense_2 = layers.Dense(num_classes, activation='sigmoid')

  def call(self, inputs): 
    x = self.dense_1(inputs)
    return self.dense_2(x)

model = MyModel(num_classes=10)

model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

@tf.function
def trace():
  data = np.random.random((1, 32))
  model(data)


logdir = "trace_log"
writer = tf.summary.create_file_writer(logdir)
tf.summary.trace_on(graph=True, profiler=True)
# Forward pass
trace()
with writer.as_default():
  tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=logdir)
#maybe works?
#https://stackoverflow.com/questions/61427583/how-do-i-plot-a-keras-tensorflow-subclassing-api-model