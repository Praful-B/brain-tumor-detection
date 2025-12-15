import tensorflow as tf

model = tf.keras.models.load_model('deMLon_model.h5')
model.summary()