import numpy as np
import matplotlib.pyplot as plt
import pandas
import os
import h5py
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
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
	model.add(Dense(nfilters[0], activation=activations[0], kernel_initializer="normal", input_dim=dim))
	for i in range(1, nlayers):
		model.add(Dense(nfilters[i], activation=activations[i]))
	model.add(Dense(1, kernel_initializer="normal"))
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
