# TensorFlow code for creating a CNN
import tensorflow as tf

from tensorflow.keras import layers, models

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(64, (3,3), activation='relu'))


model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()


