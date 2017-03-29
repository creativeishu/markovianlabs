# A sample CNN network

import os
import h5py
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.convolutional import ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import optimizers
from sys import argv, exit
from keras import backend as K

__author__ = 'irshad'

#==============================================================================

folder = argv[1]
train_data_dir = folder+'train'
validation_data_dir = folder+'validation'
savefilename = folder+'savedmodels/cnn_small.hdf5'

img_width, img_height = 224, 224
nb_epoch = 50
batch_size = 32

train_samples = 58787
valid_samples = 28919

nb_class = 15

if (K.image_data_format()=='channels_last'):
	inputshape = (img_width, img_height, 3)
elif (K.image_data_format()=='channels_first'):
	inputshape = (3, img_width, img_height)
else:
	print "Invalid  data format: ", K.image_data_format()
	exit()

#==============================================================================

print "============================================"
print "Data Format: ", K.image_data_format()
print "Input shape: ", inputshape
print "Number of classes: ", nb_class
print "============================================"

#==============================================================================

model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=inputshape))
model.add(Conv2D(64, (3, 3), activation='relu', name='conv1_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(64, (3, 3), activation='relu', name='conv1_2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(128, (3, 3), activation='relu', name='conv2_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(128, (3, 3), activation='relu', name='conv2_2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_class, activation='sigmoid'))

model.compile(loss='binary_crossentropy', \
	optimizer='adadelta', metrics=['accuracy'])

print "Model loaded and compiled"
print model.summary()

#==============================================================================

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, \
	zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_data_dir, \
	target_size=(img_width, img_height), batch_size=batch_size, class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(validation_data_dir, \
	target_size=(img_width, img_height), batch_size=batch_size, class_mode='categorical')

model.fit_generator(train_generator, validation_data=validation_generator, \
	steps_per_epoch=train_samples//batch_size, epochs=nb_epoch, \
	validation_steps=valid_samples/batch_size)
model.save(savefilename, overwrite=True)

#==============================================================================
