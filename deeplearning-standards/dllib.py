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

from sys import exit

#==============================================================================

class Deeplearninglibrary(object):
	"""
	Deeplearninglibrary
	"""

	def __init__(self):
		self.xtrain = []
		self.ytrain = []
		self.xvalid = []
		self.yvalid = []

		print "Deep Learning library is initialised"


	def loaddata(self, xtrain, ytrain, xvalid, yvalid):
		self.xtrain = xtrain
		self.ytrain = ytrain
		self.xvalid = xvalid
		self.yvalid = yvalid


	def fullyconnected(self, nlayers, nfilters, activations, dim):
		if not (len(nfilters)==nlayers and len(activations)==nlayers):
			print "Wrong input"
			exit()
		model = Sequential()
		model.add(Dense(nfilters[0], input_dim=dim, init='normal', activation=activations[0]))
		for i in range(nlayers-1):
			model.add(Dense(nfilters[i], activation=activations[i]))
		model.add(Dense(1, init='normal'))
		return model


	def train(self, model, batchsize, nepoch, loss, optimizer, \
			save=False, savefilename='weights.h5'):
		model.compile(loss=loss, optimizer=optimizer)
		model.fit(self.xtrain, self.ytrain, batch_size=batchsize, nb_epoch=nepoch, \
			verbose=1, validation_data=(self.xvalid, self.yvalid))
		if save:
			model.save_weights(savefilename, overwrite=True)


	def predict(self, model, X):
		return model.predict(X)


	def simplerun(self, nlayers=1, nf=32, act='relu', nepoch=1, batchsize=32, \
			loss='mean_squared_error', optimizer='adadelta'):
		nfilters = [nf] * nlayers
		activation = [act] * nlayers
		model = self.fullyconnected(nlayers, nfilters, activation, dim=50)
		print model.summary()
		self.train(model, batchsize, nepoch, loss, optimizer)
		return model


#==============================================================================

if __name__=="__main__":

	nlayers = 5
	nf = 128
	act = 'relu'
	nepoch = 10
	batchsize = 32
	loss = 'mean_squared_error'
	optimizer = 'adadelta'
	metrics = ['accuracy']
	inputfilename = "../irshad/photoredshifts/data/CMASS_50features_irshad.csv"
	savefilename = '../savedmodels/model_%i_%i_%s.h5'%(nlayers, nf, act)

	nfilters = [nf] * nlayers
	activation = [act] * nlayers

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

	ob = Deeplearninglibrary()
	ob.loaddata(X_train, Y_train, X_test, Y_test)
	model = ob.simplerun(nlayers, nf, act, nepoch)

	ypred = ob.predict(model, X_test)
	plt.plot(Y_test, ypred, 'xr')
	plt.plot(Y_test, Y_test, 'k')
	plt.xlim(0,1)
	plt.ylim(0,1)
	plt.show()

	# model = ob.fullyconnected(nlayers, nfilters, activation, nCol)
	# print model.summary()

	# ob.train(model, batchsize, nepoch, loss, optimizer)




