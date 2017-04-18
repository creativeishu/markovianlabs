import pickle
import numpy as np 
import matplotlib.pyplot as plt 
from sys import exit, argv 
from keras import backend as K
from keras.layers import Flatten, Dense, Dropout, Input
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model, Sequential

#==============================================================================

__author__ = "Irshad Mohammed"

#==============================================================================

from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.reshape(x_train, \
			(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
x_test = np.reshape(x_test, \
			(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

print "Training set: ", x_train.shape, y_train.shape
print "Test set: ", x_test.shape, y_test.shape

#==============================================================================

nlayers_conv = 3
conv_kernel = (3,3)
filters_conv = [128, 64, 32]
inputshape = x_train.shape[1:]
activation = 'relu'
batch_size = 1000
savefilename = 'encoder.hdf5'

#==============================================================================

model = Sequential()

model.add(Conv2D(x_train.shape[-1], conv_kernel, \
	padding='same', activation=activation, input_shape=inputshape))

for i in range(nlayers_conv):
    model.add(Conv2D(filters_conv[i], conv_kernel, \
    	padding='same', activation=activation))

for i in range(nlayers_conv):
    model.add(Conv2D(filters_conv[nlayers_conv-1-i], conv_kernel, \
    	padding='same', activation=activation))

model.add(Conv2D(x_train.shape[-1], conv_kernel, \
	padding='same', activation=activation))

model.compile(optimizer='rmsprop', loss='mean_squared_error')

print model.summary()

#==============================================================================

hist = model.fit(x_train, x_train, epochs=50, batch_size=batch_size)
model.save(savefilename)
pickle.dump(hist.history, open(savefilename.replace('.hdf5', '_hist.p'), "w"))
