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
nf = 128
act = 'relu'
nepoch = 40
batchsize = 512
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

model = Sequential()
model.add(Dense(nf, activation='relu', \
	kernel_initializer="normal", input_dim=nCol))
model.add(Dense(nf, kernel_initializer="normal", activation='relu'))
model.add(Dense(nf, kernel_initializer="normal", activation='relu'))
model.add(Dense(1, kernel_initializer="normal"))
model.compile(loss=loss, optimizer=optimizer)
print model.summary()
model.fit(X_train, Y_train, batch_size=batchsize, epochs=nepoch, verbose=1, \
	validation_split=val_split)

#==============================================================================

nn = len(model.layers)
model.layers.pop()
for layer in model.layers[:nn]:
    layer.trainable = False
model.add(Dense(nf, kernel_initializer="normal", activation='relu'))
model.add(Dense(nf, kernel_initializer="normal", activation='relu'))
model.add(Dense(1, kernel_initializer="normal"))
model.compile(loss=loss, optimizer=optimizer)
print model.summary()
model.fit(X_train, Y_train, batch_size=batchsize, epochs=nepoch, verbose=1, \
	validation_split=val_split)

#==============================================================================

nn = len(model.layers)
model.layers.pop()
for layer in model.layers[:nn]:
    layer.trainable = False
model.add(Dense(nf, kernel_initializer="normal", activation='relu'))
model.add(Dense(nf, kernel_initializer="normal", activation='relu'))
model.add(Dense(1, kernel_initializer="normal"))
model.compile(loss=loss, optimizer=optimizer)
print model.summary()
model.fit(X_train, Y_train, batch_size=batchsize, epochs=nepoch, verbose=1, \
	validation_split=val_split)

#==============================================================================

nn = len(model.layers)
model.layers.pop()
for layer in model.layers[:nn]:
    layer.trainable = False
model.add(Dense(nf, kernel_initializer="normal", activation='relu'))
model.add(Dense(nf, kernel_initializer="normal", activation='relu'))
model.add(Dense(1, kernel_initializer="normal"))
model.compile(loss=loss, optimizer=optimizer)
print model.summary()
model.fit(X_train, Y_train, batch_size=batchsize, epochs=nepoch, verbose=1, \
	validation_split=val_split)

#==============================================================================

nn = len(model.layers)
model.layers.pop()
for layer in model.layers[:nn]:
    layer.trainable = False
model.add(Dense(nf, kernel_initializer="normal", activation='relu'))
model.add(Dense(nf, kernel_initializer="normal", activation='relu'))
model.add(Dense(1, kernel_initializer="normal"))
model.compile(loss=loss, optimizer=optimizer)
print model.summary()
model.fit(X_train, Y_train, batch_size=batchsize, epochs=nepoch, verbose=1, \
	validation_split=val_split)

#==============================================================================

model.save(savefilename, overwrite=True)

#==============================================================================
