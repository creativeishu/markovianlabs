import pickle
import numpy as np 
import matplotlib.pyplot as plt 
from sys import exit, argv 
from keras import backend as K
from keras.layers import Flatten, Dense, Dropout, Input
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model, Sequential
from keras.datasets import cifar100

#==============================================================================

__author__ = "Irshad Mohammed"

#==============================================================================

Xtrain = np.load('data162_lens.npy')
Ytrain = np.load('data161_lens.npy')

# Xtrain = np.load('/data/mohammed/data/deeplensing/data162/data162_lens.npy')
# Ytrain = np.load('/data/mohammed/data/deeplensing/data161/data161_lens.npy')

Xtrain = Xtrain/255.0
Ytrain = Ytrain/255.0

nsamples_train = 10000
nsamples_valid = 10000

xtrain = Xtrain[:nsamples_train]
xvalid = Xtrain[nsamples_train:nsamples_train+nsamples_valid]

ytrain = Ytrain[:nsamples_train]
yvalid = Ytrain[nsamples_train:nsamples_train+nsamples_valid]

print "Training set: ", xtrain.shape, ytrain.shape
print "Test set: ", xvalid.shape, yvalid.shape
# exit()
#==============================================================================

nlayers_conv = 3
conv_kernel = (3,3)
filters_conv = [128, 64, 32]
inputshape = xtrain.shape[1:]
outputshape = ytrain.shape[1:]
activation = 'relu'
batch_size = 100
savefilename = 'denoising_162_161.hdf5'

print "Inputshape: ", inputshape
print "outputshape: ", outputshape
# exit()
#==============================================================================

# model = Sequential()

# model.add(Conv2D(xtrain.shape[-1], conv_kernel, \
# 	padding='same', activation=activation, input_shape=inputshape))

# for i in range(nlayers_conv):
#     model.add(Conv2D(filters_conv[i], conv_kernel, \
#     	padding='same', activation=activation))

# for i in range(nlayers_conv):
#     model.add(Conv2D(filters_conv[nlayers_conv-1-i], conv_kernel, \
#     	padding='same', activation=activation))

# model.add(Conv2D(outputshape[-1], conv_kernel, \
# 	padding='same', activation=activation))

# model.compile(optimizer='rmsprop', loss='mean_squared_error')

# print model.summary()
# exit()
#==============================================================================

input_img = Input(shape=inputshape)  # adapt this if using `channels_first` image data format
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (7, 7, 32)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(outputshape[-1], (3, 3), activation='sigmoid', padding='same')(x)

model = Model(input_img, decoded)
model.compile(optimizer='adadelta', loss='binary_crossentropy')
print model.summary()

#==============================================================================

hist = model.fit(xtrain, ytrain, \
					epochs=50, batch_size=batch_size, \
					validation_data=(xvalid, yvalid))
model.save(savefilename)
pickle.dump(hist.history, open(savefilename.replace('.hdf5', '_hist.p'), "w"))
