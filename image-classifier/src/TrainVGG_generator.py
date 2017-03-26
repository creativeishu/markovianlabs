"""
Training the VGG16 or VGG19 architectures with new data
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD
from keras import backend as K

from sys import exit, argv
__author__ = 'irshad'

#==============================================================================

def VGG16(inputshape, nb_class=2):
	"""
	Loads and builds the VGG16 model for object recognition.

	"""
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

	model.add(ZeroPadding2D((1, 1)))
	model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_1'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_2'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_3'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))

	model.add(ZeroPadding2D((1, 1)))
	model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_1'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_2'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_3'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))

	model.add(ZeroPadding2D((1, 1)))
	model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_1'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_2'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_3'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))

	model.add(Flatten())
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(nb_class, activation='softmax'))

	return model

#==============================================================================

def VGG19(inputshape, nb_class=2):
	"""
	Loads and builds the VGG19 model for object recognition.

	"""
	model = Sequential()
	model.add(ZeroPadding2D((1,1), input_shape=inputshape))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(256, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(256, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(256, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(256, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(Flatten())
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(nb_class, activation='softmax'))

	return model

#==============================================================================

def trainvgg(path_train, path_valid, train_samples, valid_samples, vgg='vgg19', \
		batch_size=32, nb_class=2, nb_epoch=1, verbose=1, \
		loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'], \
		save=False, savefilename='weights.hdf5'):

	img_width, img_height = 150, 150
	if (K.image_data_format()=='channels_last'):
		inputshape = (img_width, img_height, 3)
	elif (K.image_data_format()=='channels_first'):
		inputshape = (3, img_width, img_height)
	else:
		print "Invalid  data format: ", K.image_data_format()
		exit()

	print "============================================"
	print "Data Format: ", K.image_data_format()
	print "Input shape: ", inputshape
	print "Number of classes: ", nb_class
	print "============================================"

	if vgg=='vgg19':
		model = VGG19(inputshape, nb_class)
	elif vgg=='vgg16':
		model = VGG16(inputshape, nb_class)
	else:
		print "Enter a valid vgg model: either vgg16 or vgg19"
		exit()

	print model.summary()
	sgd = optimizers.SGD(lr=0.01, momentum=0.9, decay=1e-6)

	model.compile(loss=loss, optimizer=sgd, metrics=metrics)

	train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, \
		zoom_range=0.2, horizontal_flip=True)
	test_datagen = ImageDataGenerator(rescale=1./255)

	train_generator = train_datagen.flow_from_directory(path_train, \
		target_size=(img_width, img_height), batch_size=batch_size, \
		class_mode='categorical')

	validation_generator = test_datagen.flow_from_directory(path_valid, \
		target_size=(img_width, img_height), batch_size=batch_size, \
		class_mode='categorical')

	model.fit_generator(train_generator, validation_data=validation_generator, \
		steps_per_epoch=train_samples//batch_size, epochs=nb_epoch, \
		validation_steps=valid_samples/batch_size)

	if save:
		model.save(savefilename, overwrite=True)
	return model

#==============================================================================

folder = argv[1]
train_data_dir = folder+'train'
validation_data_dir = folder+'validation'
savefilename = folder+'savedmodels/firstattempt_vgg_generator.hdf5'

vgg = 'vgg19'
finalmodel = trainvgg(train_data_dir, validation_data_dir, \
	58787, 28919, vgg=vgg, nb_class=15, batch_size=1024, nb_epoch=1, \
	save=True, savefilename=savefilename)

#==============================================================================
