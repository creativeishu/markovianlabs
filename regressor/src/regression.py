import pandas
import numpy as np 
from sys import exit

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import ZeroPadding1D

__author__ = "Irshad Mohammed"

#==============================================================================

class regression(object):
	"""
	main doc string
	"""
	def __init__(self, filename, shuffle=True):
		"""
		constructor's doc string
		"""
		dataframe = pandas.read_csv(filename)
		self.dataset = dataframe.values
		if shuffle:
			np.random.shuffle(self.dataset)
		print "Class is initialised"

#------------------------------------------------------------------------------

	def make_data(self, xcols, ycol, nb_train_samples, nb_valid_samples=None):
		self.nb_train_samples = nb_train_samples
		self.nb_valid_samples = nb_valid_samples

		self.X_train = self.dataset[:nb_train_samples, xcols]
		self.Y_train = self.dataset[:nb_train_samples, ycol]
		self.X_train = self.X_train.reshape(self.X_train.shape[0], \
											self.X_train.shape[1], 1)
		print "Training dataset: ", self.X_train.shape, self.Y_train.shape

		if nb_valid_samples != None:
			self.X_valid = self.dataset[nb_train_samples:\
									nb_train_samples+nb_valid_samples, xcols]
			self.Y_valid = self.dataset[nb_train_samples:\
									nb_train_samples+nb_valid_samples, ycol]
			self.X_valid = self.X_valid.reshape(self.X_valid.shape[0], \
									self.X_valid.shape[1], 1)
			print "Validation dataset: ", self.X_valid.shape, \
									self.Y_valid.shape
		self.input_shape = self.X_train.shape[1:]
		print "Input shape: ", self.input_shape

#------------------------------------------------------------------------------

	def make_network(self, ncov=2, ndense=2, \
					nfilters_cov=128, nfilters_dense=128, \
					kernel_size=2, activation_cov='relu', \
					activation_dense='relu'):
		self.ncov = ncov
		self.ndense = ndense
		if type(nfilters_cov)==int:
			self.nfilters_cov = [nfilters_cov]*self.ncov
		elif type(nfilters_cov)==list:
			self.nfilters_cov = nfilters_cov
		else:
			print "Provide an integer or list for nfilter_cov"
			exit()

		if type(nfilters_dense)==int:
			self.nfilters_dense = [nfilters_dense]*self.ndense
		elif type(nfilters_dense)==list:
			self.nfilters_dense = nfilters_dense
		else:
			print "Provide an integer or list for nfilter_dense"
			exit()

		self.model = Sequential()
		if self.ncov>0:
			self.model.add(Conv1D(self.nfilters_cov[0], kernel_size, \
					padding='valid', \
                  	input_shape=self.input_shape, \
                  	activation=activation_cov))
			for i in range(1, self.ncov):
				self.model.add(Conv1D(self.nfilters_cov[i], kernel_size, \
										activation=activation_cov))
		else:
			self.model.add(ZeroPadding1D(padding=1, \
						input_shape=self.input_shape))

		self.model.add(Flatten())
		for i in range(ndense):
			self.model.add(Dense(self.nfilters_dense[i], \
									activation=activation_dense))
			self.model.add(Dropout(0.5))
		self.model.add(Dense(1))

#------------------------------------------------------------------------------
	
	def train_network(self, ncov=5, ndense=3, \
					nfilters_cov=128, nfilters_dense=128, \
					kernel_size=2, \
					activation_cov='relu', activation_dense='relu', \
					loss='mean_squared_error', optimizer='adadelta', \
					batch_size=32, nb_epochs=100, verbose=1, \
					save=True, savefilename='model_regression.hdf5'):

		self.make_network(ncov, ndense, \
					nfilters_cov, nfilters_dense, \
					kernel_size, activation_cov, activation_dense)
		self.model.compile(loss=loss, optimizer=optimizer)
		print self.model.summary()

		if self.nb_valid_samples != None:
			self.hist = self.model.fit(self.X_train, self.Y_train, \
					batch_size=batch_size, epochs=nb_epochs, \
					verbose=verbose, \
					validation_data=(self.X_valid, self.Y_valid))
		else:
			self.hist = self.model.fit(self.X_train, self.Y_train, \
					batch_size=batch_size, epochs=nb_epochs, \
					verbose=verbose)
		if save:
			self.model.save(savefilename)
		return self.hist

#==============================================================================

if __name__ == "__main__":
	filename = '/Users/mohammed/Dropbox/fermilabwork/with_myself/\
photoredshifts/data/CMASS_50features_irshad.csv'
	obj = regression(filename)
	xcols = tuple(range(1, 51))
	ycol = 51
	obj.make_data(xcols, ycol, 25000)
	obj.train_network(ncov=0, ndense=3, nb_epochs=100)
