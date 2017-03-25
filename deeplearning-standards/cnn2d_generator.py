# A sample CNN network

import os
import h5py
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
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
savefilename = folder+'savedmodels/firstattempt_generator.hdf5'

img_width, img_height = 150, 150
nb_epoch = 150
batch_size = 64

train_samples = 58787
valid_samples = 28919


# nb_class = 2
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
# print "Number of classes: ", nb_class
print "============================================"

#==============================================================================

simplemodel = Sequential()
simplemodel.add(Conv2D(32, (3, 3), input_shape=inputshape, activation='relu'))
simplemodel.add(MaxPooling2D(pool_size=(2, 2)))

simplemodel.add(Conv2D(32, (3, 3), activation='relu'))
simplemodel.add(MaxPooling2D(pool_size=(2, 2)))

simplemodel.add(Conv2D(64, (3, 3), activation='relu'))
simplemodel.add(MaxPooling2D(pool_size=(2, 2)))

simplemodel.add(Flatten())
simplemodel.add(Dense(64, activation='relu'))
simplemodel.add(Dropout(0.5))
simplemodel.add(Dense(1, activation='sigmoid'))

simplemodel.compile(loss='binary_crossentropy', \
	optimizer='adadelta', metrics=['accuracy'])

print "Model loaded and compiled"
print simplemodel.summary()

#==============================================================================

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, \
	zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_data_dir, \
	target_size=(img_width, img_height), batch_size=batch_size, class_mode='binary')

validation_generator = test_datagen.flow_from_directory(validation_data_dir, \
	target_size=(img_width, img_height), batch_size=batch_size, class_mode='binary')

simplemodel.fit_generator(train_generator, validation_data=validation_generator, \
	steps_per_epoch=train_samples//batch_size, epochs=nb_epoch, \
	validation_steps=valid_samples/batch_size)
simplemodel.save(savefilename, overwrite=True)

#==============================================================================
