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
from keras.datasets import cifar100

#==============================================================================

__author__ = "Irshad Mohammed"

#==============================================================================

from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
x_train = x_train/255.0
x_test = x_test/255.0

xtrain = x_train[:,:,:,:2]
ytrain = x_train[:,:,:,2]

xtest = x_test[:,:,:,:2]
ytest = x_test[:,:,:,2]

ytrain = np.reshape(ytrain, \
	(ytrain.shape[0], ytrain.shape[1], ytrain.shape[2], 1))
ytest = np.reshape(ytest, \
	(ytest.shape[0], ytest.shape[1], ytest.shape[2], 1))

print "Training set: ", xtrain.shape, ytrain.shape
print "Test set: ", xtest.shape, ytest.shape

#==============================================================================

nlayers_conv = 3
conv_kernel = (3,3)
filters_conv = [128, 64, 32]
inputshape = xtrain.shape[1:]
outputshape = ytrain.shape[1:]
activation = 'relu'
batch_size = 1000
savefilename = 'encoder.hdf5'

print "Inputshape: ", inputshape
print "outputshape: ", outputshape

#==============================================================================

model = Sequential()

model.add(Conv2D(xtrain.shape[-1], conv_kernel, \
	padding='same', activation=activation, input_shape=inputshape))

for i in range(nlayers_conv):
    model.add(Conv2D(filters_conv[i], conv_kernel, \
    	padding='same', activation=activation))

for i in range(nlayers_conv):
    model.add(Conv2D(filters_conv[nlayers_conv-1-i], conv_kernel, \
    	padding='same', activation=activation))

model.add(Conv2D(outputshape[-1], conv_kernel, \
	padding='same', activation=activation))

model.compile(optimizer='rmsprop', loss='mean_squared_error')

print model.summary()

#==============================================================================

hist = model.fit(xtrain, ytrain, \
					epochs=50, batch_size=batch_size, \
					validation_data=(xtest, ytest))
model.save(savefilename)
pickle.dump(hist.history, open(savefilename.replace('.hdf5', '_hist.p'), "w"))
