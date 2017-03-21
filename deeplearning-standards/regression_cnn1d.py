import numpy as np
import matplotlib.pyplot as plt
import pandas
import os
import h5py
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers.convolutional import Conv1D
from keras.layers import Activation, Dropout, Flatten, Dense

np.random.seed(1234)

from sys import exit, argv

#==============================================================================

folder=argv[1]
nlayers = 5
nf = 128
act = 'relu'
nepoch = 100
batchsize = 64
loss = 'mean_squared_error'
optimizer = 'adadelta'
inputfilename = folder+"/CMASS_50features_irshad.csv"
savefilename = folder+'/model_%i_%i_%s.hdf5'%(nlayers, nf, act)
val_split = 0.5
kernel_size = 2

#==============================================================================

filename = inputfilename
dataframe = pandas.read_csv(filename)
dataset = dataframe.values

nCol = 50
X_train = dataset[:,1:nCol+1]
Y_train = dataset[:,nCol+1]

print "Shape of X-train: ", X_train.shape
print "Shape of Y-train: ", Y_train.shape

#==============================================================================

def mymmodel(nlayers, nfilters, activations, dim):
	if not (len(nfilters)==nlayers and len(activations)==nlayers):
		print "Wrong input"
		exit()

	model = Sequential()
	model.add(Conv1D(nfilters[0], kernel_size, padding='valid', input_shape=(kernel_size, dim), activation='relu'))
	print model.output_shape
	model.add(Conv1D(nfilters[0], kernel_size, activation='relu'))
	print model.output_shape
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1))	
	return model

nfilters = [nf] * nlayers
activation = [act] * nlayers
model = mymmodel(nlayers, nfilters, activation, X_train.shape[1])
print model.summary()

#==============================================================================

model.compile(loss=loss, optimizer=optimizer)
model.fit(X_train, Y_train, batch_size=batchsize, epochs=nepoch, verbose=1, \
	validation_split=val_split)
model.save(savefilename, overwrite=True)

#==============================================================================
