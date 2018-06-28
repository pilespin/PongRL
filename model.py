
import tensorflow as tf
from tfHelper import tfHelper

k = tf.keras

def model():
	model = k.models.Sequential()
	model.add(k.layers.Conv2D(1, (10, 10), activation='relu',
	                 input_shape = (160, 210, 3)))
	# model.add(k.layers.BatchNormalization())
	model.add(k.layers.Conv2D(1, (10, 10), activation='relu', padding='same'))
	# model.add(k.layers.BatchNormalization())
	model.add(k.layers.MaxPool2D(strides=(5,5)))
	# model.add(k.layers.Dropout(0.25))
	# model.add(k.layers.Conv2D(1, (10, 10), activation='relu', padding='same'))
	# model.add(k.layers.MaxPool2D(strides=(10,10)))

	model.add(k.layers.Flatten())
	# model.add(k.layers.Dense(1000, activation='relu'))
	model.add(k.layers.Dense(100, activation='relu'))
	# model.add(k.layers.Dropout(0.2))
	model.add(k.layers.Dense(6, activation='softmax'))

	return model

