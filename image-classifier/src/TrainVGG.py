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

def trainvgg(xtrain, ytrain, validation_split=0.4, vgg='vgg19', \
		batch_size=32, nb_epoch=1, verbose=1, \
		loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'], \
		save=False, savefilename='weights.hdf5'):

	inputshape = tuple(xtrain.shape[1:])
	nb_class = ytrain.shape[1]

	print "Training set shape: ", xtrain.shape, ytrain.shape

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
	model.fit(xtrain, ytrain, batch_size=batch_size, epochs=nb_epoch, \
		verbose=verbose, validation_split=validation_split)

	if save:
		model.save(savefilename, overwrite=True)
	return model

#==============================================================================

folder = argv[1]
xdata = np.load(folder+'xdata.npy')
ydata = np.load(folder+'ydata.npy')
#xdata = np.transpose(xdata, (0, 2,3,1))
xdata = xdata.astype('float32')
xdata /= 255

vgg = 'vgg19'
savefile = folder+'savedmodels/firstmodel.hdf5'
finalmodel = trainvgg(xdata, ydata, vgg=vgg, nb_epoch=50, \
	validation_split=0.5, save=True, savefilename=savefile)

#==============================================================================
