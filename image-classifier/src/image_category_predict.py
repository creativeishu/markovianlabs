"""
Models for object recognition. Based on VGG16 and VGG19.
"""

import os
import h5py
import cv2
import sys
import numpy as np
import json


from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD

# from keras import backend as K
# K.set_image_dim_ordering('th')



__author__ = 'jverma'



def VGG16(weights_path, inputshape):
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
	model.add(ZeroPadding2D((1, 1), input_shape=inputshape))
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
	model.add(Dense(1000, activation='softmax'))

	model.load_weights(weights_path)
	return model




def VGG19(weights_path, inputshape):
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
	model.add(Dense(1000, activation='softmax'))

	model.load_weights(weights_path)
	return model





def category_predict(input_image, model_type='vgg16', img_width=224, img_height=224, k=5):
	"""
	Prints top categories for the input image.

	Parameters
	----------
	input_image: The image to be recognized.
	model_type: Type of model to be used, either vgg16 or vgg19.
	img_width: Width of the image.
	img_height: Height of the image.
	k: Number of categories to be returned.
	"""
	path = '/Users/%s/Dropbox/irshad2janu/deeplearning_datasets/image_classifiers/vggfiles/'%os.getlogin()

	class_dict = {}
	index_file = open(path+'synset_words.txt')
	for i,line in enumerate(index_file):
		record = line.rstrip().split(' ')
		class_dict[i] = record[1:]

	im = cv2.resize(cv2.imread(input_image), (224, 224)).astype(np.float32)
	# im[:,:,0] -= 103.939
	# im[:,:,1] -= 116.779
	# im[:,:,2] -= 123.68
	im = im.transpose((2,0,1))
	im = np.expand_dims(im, axis=0)
	inputshape = tuple(im.shape[1:])
	print "Input Shape: ", inputshape

	if (model_type == 'vgg16'):
		weights_path = path+'vgg16_weights.h5'
		model = VGG16(weights_path, inputshape)
	elif (model_type == 'vgg19'):
		weights_path = path+'vgg19_weights.h5'
		model = VGG19(weights_path, inputshape)
	else:
		print "Choose model_type vgg16 or vgg19."
		return None

	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optimizer=sgd, loss='categorical_crossentropy')
	out = model.predict_proba(im)

	idxs = np.argsort(out[0])[::-1][:k]
	for x in idxs:
		print(class_dict[x]), out[0][x]
	return None





input_image = sys.argv[1]
category_predict(input_image, model_type='vgg19')




