"""
Models for object recognition. Based on VGG16 and VGG19.
"""

import os
import h5py
import cv2
import sys
import numpy as np
import json
from glob import glob
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD
from keras.utils import np_utils

from keras import backend as K
K.set_image_dim_ordering('th')


from sys import exit
__author__ = 'irshad'

#==============================================================================

def VGG16(inputshape, nb_class=2):
	"""
	Loads and builds the VGG16 model for object recognition.

	Parameters
	----------
	weights_path: path to the HD5 file containing the weights of the pre-trained model.
	img_width: Width of the image.
	img_height: Height of the image.

	Returns
	-------
	pre-trained model.
	"""
	model = Sequential()
	model.add(ZeroPadding2D((1,1),input_shape=inputshape))
	model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))

	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))

	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))

	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))

	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
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

	Parameters
	----------
	weights_path: path to the HD5 file containing the weights of the pre-trained model.
	img_width: Width of the image.
	img_height: Height of the image.

	Returns
	-------
	pre-trained model.
	"""
	model = Sequential()
	model.add(ZeroPadding2D((1,1),input_shape=inputshape))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(Flatten())
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(nb_class, activation='softmax'))

	return model

#==============================================================================

def trainvgg(xtrain, ytrain, xvalid, yvalid, vgg='vgg19', \
		batch_size=32, nb_epoch=1, verbose=1, \
		loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'], \
		save=False, savefilename='weights.h5'):

	inputshape = tuple(xtrain.shape[1:])
	nb_class = ytrain.shape[1]

	print "Training set shape: ", xtrain.shape, ytrain.shape
	print "Validation set shape: ", xvalid.shape, yvalid.shape

	if vgg=='vgg19':
		model = VGG19(inputshape, nb_class)
	elif vgg=='vgg16':
		model = VGG16(inputshape, nb_class)
	else:
		print "Enter a valid vgg model: either vgg16 or vgg19"
		exit()

	print model.summary()

	model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
	model.fit(xtrain, ytrain, batch_size=batch_size, nb_epoch=nb_epoch, \
		verbose=verbose, validation_data=(xvalid, yvalid))

	if save:
		model.save_weights(savefilename, overwrite=True)
	return model

#==============================================================================

xtrain = np.load('../data/catsdogs/xtrain.npy')
ytrain = np.load('../data/catsdogs/ytrain.npy')
xvalid = np.load('../data/catsdogs/xvalid.npy')
yvalid = np.load('../data/catsdogs/yvalid.npy')

vgg = 'vgg16'
finalmodel = trainvgg(xtrain, ytrain, xvalid, yvalid, \
	vgg=vgg, save=False)

#==============================================================================

# traincat = glob('../data/catsdogs/train/cats/cat.*')
# traindog = glob('../data/catsdogs/train/dogs/dog.*')
# validationcat = glob('../data/catsdogs/validation/cats/cat.*')
# validationdog = glob('../data/catsdogs/validation/dogs/dog.*')


# folder = '../data/catsdogs/'
# xtrain = []
# ytrain = []
# xvalid = []
# yvalid = []

# for i in range(len(traincat)):
# 	input_image = traincat[i]
# 	im = cv2.resize(cv2.imread(input_image), (150, 150)).astype(np.float32)
# 	im = np.transpose(im)
# 	im[0] = np.transpose(im[0])
# 	xtrain.append(im)
# 	ytrain.append(0)

# 	input_image = traindog[i]
# 	im = cv2.resize(cv2.imread(input_image), (150, 150)).astype(np.float32)
# 	im = np.transpose(im)
# 	im[0] = np.transpose(im[0])
# 	xtrain.append(im)
# 	ytrain.append(1)

# for i in range(len(validationcat)):
# 	input_image = validationcat[i]
# 	im = cv2.resize(cv2.imread(input_image), (150, 150)).astype(np.float32)
# 	im = np.transpose(im)
# 	im[0] = np.transpose(im[0])
# 	xvalid.append(im)
# 	yvalid.append(0)

# 	input_image = validationdog[i]
# 	im = cv2.resize(cv2.imread(input_image), (150, 150)).astype(np.float32)
# 	im = np.transpose(im)
# 	im[0] = np.transpose(im[0])
# 	xvalid.append(im)
# 	yvalid.append(1)

# xtrain = np.array(xtrain)
# ytrain = np.array(ytrain)
# xvalid = np.array(xvalid)
# yvalid = np.array(yvalid)


# ind_train = np.arange(len(xtrain))
# np.random.shuffle(ind_train)

# ind_validation = np.arange(len(xvalid))
# np.random.shuffle(ind_validation)

# xtrain = xtrain[ind_train]
# ytrain = ytrain[ind_train]

# xvalid = xvalid[ind_validation]
# yvalid = yvalid[ind_validation]


# # One Hot Encoding
# def one_hot_encode_object_array(arr):
#     '''One hot encode a numpy array of objects (e.g. strings)'''
#     uniques, ids = np.unique(arr, return_inverse=True)
#     return np_utils.to_categorical(ids, len(uniques))

# # One hot encode labels for training and test sets.
# ytrain = one_hot_encode_object_array(ytrain)
# yvalid = one_hot_encode_object_array(yvalid)

# print xtrain.shape, ytrain.shape
# print xvalid.shape, yvalid.shape

# np.save('../data/catsdogs/xtrain.npy', xtrain)
# np.save('../data/catsdogs/ytrain.npy', ytrain)
# np.save('../data/catsdogs/xvalid.npy', xvalid)
# np.save('../data/catsdogs/yvalid.npy', yvalid)
# exit()
