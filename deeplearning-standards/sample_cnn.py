import os
import h5py
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import optimizers
from sys import argv

folder = argv[1]
X_train = np.load(folder+'xdata.npy')
Y_train = np.load(folder+'ydata.npy')
print X_train.shape, Y_train.shape

inputshape = X_train.shape[1:]
nclass = Y_train.shape[1]
print "shape: ", inputshape, nclass

simplemodel = Sequential()
simplemodel.add(Convolution2D(32, 3, 3, input_shape=inputshape))
simplemodel.add(Activation('relu'))
simplemodel.add(MaxPooling2D(pool_size=(2, 2)))

simplemodel.add(Convolution2D(32, 3, 3))
simplemodel.add(Activation('relu'))
simplemodel.add(MaxPooling2D(pool_size=(2, 2)))

simplemodel.add(Convolution2D(64, 3, 3))
simplemodel.add(Activation('relu'))
simplemodel.add(MaxPooling2D(pool_size=(2, 2)))

simplemodel.add(Flatten())
simplemodel.add(Dense(64))
simplemodel.add(Activation('relu'))
simplemodel.add(Dropout(0.5))
simplemodel.add(Dense(nclass))
simplemodel.add(Activation('sigmoid'))

simplemodel.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
print simplemodel.summary()
simplemodel.fit(X_train, Y_train, batch_size=32, nb_epoch=1, \
	verbose=1, validation_split=0.4)