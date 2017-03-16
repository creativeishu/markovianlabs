import numpy as np
import matplotlib.pyplot as plt
import pandas
import os
import h5py
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution1D, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense

# Forcing it to use theono as the backend
#from keras import backend as K
#K.set_image_dim_ordering('th')

np.random.seed(1234)

from sys import exit, argv

#==============================================================================

folder=argv[1]
nlayers = 6
nf = 128
act = 'relu'
nepoch = 100
batchsize = 32
loss = 'mean_squared_error'
optimizer = 'adadelta'
metrics = ['accuracy']
inputfilename = folder+"/CMASS_50features_irshad.csv"
savefilename = folder+'/model_%i_%i_%s.h5'%(nlayers, nf, act)


#==============================================================================

filename = inputfilename
dataframe = pandas.read_csv(filename)
dataset = dataframe.values

nCol = 50
nn = int(len(dataset)*2/3)
X_train = dataset[:nn,1:nCol+1]
Y_train = dataset[:nn,nCol+1]
X_test = dataset[nn:,1:nCol+1]
Y_test = dataset[nn:,nCol+1]

print "Shape of X-train: ", X_train.shape
print "Shape of Y-train: ", Y_train.shape
print "Shape of X-test: ", X_test.shape
print "Shape of Y-test: ", Y_test.shape

#==============================================================================

def mymmodel(nlayers, nfilters, activations, dim):
	if not (len(nfilters)==nlayers and len(activations)==nlayers):
		print "Wrong input"
		exit()
	model = Sequential()
	model.add(Dense(nCol, input_dim=dim, init='normal', activation=activations[0]))
	for i in range(nlayers):
		model.add(Dense(nfilters[i], activation=activations[i]))
	model.add(Dense(1, init='normal'))
	return model

nfilters = [nf] * nlayers
activation = [act] * nlayers
model = mymmodel(nlayers, nfilters, activation, X_train.shape[1])
print model.summary()

#==============================================================================
#mean_absolute_percentage_error, mean_squared_error
model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
model.fit(X_train, Y_train, batch_size=batchsize, nb_epoch=nepoch, verbose=1, \
	validation_data=(X_test, Y_test))
model.save(savefilename, overwrite=True)
#==============================================================================
