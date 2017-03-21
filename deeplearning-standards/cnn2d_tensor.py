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
X_train = np.load(folder+'xdata.npy')
Y_train = np.load(folder+'ydata.npy')
savefilename = folder+'savedmodels/firstattempt.hdf5'

#==============================================================================

if (K.image_data_format()=='channels_last') and X_train.shape[1]==3:
	X_train = np.transpose(X_train, (0, 2, 3, 1))
elif (K.image_data_format()=='channels_first') and X_train.shape[3]==3:
	X_train = np.transpose(X_train, (0, 3, 1, 2))

inputshape = X_train.shape[1:]
nclass = Y_train.shape[1]
X_train = X_train.astype('float32')
X_train /= 255.0

print "============================================"
print "Data Format: ", K.image_data_format()
print "Data shape", X_train.shape, Y_train.shape
print "Input shape: ", inputshape
print "Number of classes: ", nclass
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
simplemodel.add(Dense(nclass, activation='sigmoid'))

#==============================================================================

simplemodel.compile(loss='binary_crossentropy', \
	optimizer='adadelta', metrics=['accuracy'])

print "Model loaded and compiled"
print simplemodel.summary()

#==============================================================================

simplemodel.fit(X_train, Y_train, batch_size=32, \
	epochs=50, verbose=1, validation_split=0.2)
simplemodel.save(savefilename, overwrite=True)

#==============================================================================
