import os
import h5py
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import optimizers
from sys import argv

folder = argv[1]
X_train = np.load(folder+'xdata.npy')
Y_train = np.load(folder+'ydata.npy')
X_train = np.transpose(X_train, (0,2,3,1))

X_train = X_train.astype('float32')
X_train /= 255

inputshape = X_train.shape[1:]
nclass = Y_train.shape[1]
print "Data shape", X_train.shape, Y_train.shape
print "Input shape and classes: ", inputshape, nclass

simplemodel = Sequential()
simplemodel.add(Conv2D(32, (3, 3), input_shape=inputshape))
simplemodel.add(Activation('relu'))
simplemodel.add(MaxPooling2D(pool_size=(2, 2)))

simplemodel.add(Conv2D(32, (3, 3)))
simplemodel.add(Activation('relu'))
simplemodel.add(MaxPooling2D(pool_size=(2, 2)))

simplemodel.add(Conv2D(64, (3, 3)))
simplemodel.add(Activation('relu'))
simplemodel.add(MaxPooling2D(pool_size=(2, 2)))

simplemodel.add(Flatten())
simplemodel.add(Dense(64))
simplemodel.add(Activation('relu'))
simplemodel.add(Dropout(0.5))
simplemodel.add(Dense(nclass))
simplemodel.add(Activation('sigmoid'))

simplemodel.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

print "Model loaded and compiled"

simplemodel.fit(X_train, Y_train, batch_size=32, epochs=100, verbose=1, validation_split=0.4)
