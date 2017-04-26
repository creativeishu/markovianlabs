"""
Fitting sequential data
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sys import exit

#==============================================================================

__author__ = "Irshad Mohammed"

#==============================================================================

class lstm_sequence(object):
	"""
	main doc string
	"""
	def __init__(self, nSequence=1, sequence_length=50, \
						nlayers_lstm=2, nlayers_dense=2, \
						nfilters_lstm=32, nfilters_dense=32, 
						loss='mse', optimizer='adadelta', \
						save=True, savemodelname='model.hdf5'):
		"""
		doc string
		"""
		self.nSequence = nSequence
		self.sequence_length = sequence_length
		self.nlayers_lstm = nlayers_lstm
		self.nlayers_dense = nlayers_dense
		self.loss = loss
		self.optimizer = optimizer
		self.save = save
		self.savemodelname = savemodelname

		if type(nfilters_lstm)==int:
			self.nfilters_lstm = [nfilters_lstm]*self.nlayers_lstm
		elif type(nfilters_lstm)==list:
			self.nfilters_lstm = nfilters_lstm
		else:
			print "nfilters_lstm can be an int or a list"
			exit()

		if type(nfilters_dense)==int:
			self.nfilters_dense = [nfilters_dense]*self.nlayers_dense
		elif type(nfilters_dense)==list:
			self.nfilters_dense = nfilters_dense
		else:
			print "nfilters_dense can be an int or a list"
			exit()

		self.model = self.build_model(loss, optimizer)

#------------------------------------------------------------------------------

	def print_metadata(self):
		print "Number of sequences: ", self.nSequence
		print "Sequence length: ", self.sequence_length
		print "Number of LSTM layers: ", self.nlayers_lstm
		print "LSTM filters: ", self.nfilters_lstm
		print "Number of Dense layers: ", self.nlayers_dense
		print "Dense layer filters: ", self.nfilters_dense
		print "Loss function: ", self. loss 
		print "optimizer: ", self.optimizer
		print "Data shape: ", self.xtrain.shape, self.ytrain.shape
		print "Validatinon split: ", self.val_split
		print "Number of training samples: ", \
						int((1.0 - self.val_split)*len(self.xtrain))
		print "Number of Validation samples: ", \
						int((self.val_split)*len(self.xtrain))
		print "Batch size: ", self.batch_size
		print "Number of epochs: ", self.epochs
		if self.save:
			print "Model will be saved at: ", self.savemodelname

#------------------------------------------------------------------------------

	def build_model(self, loss='mse', optimizer='rmsprop'):
	    model = Sequential()
	    model.add(LSTM(self.nfilters_lstm[0] , \
	    				input_shape=(self.sequence_length-1, self.nSequence), \
	                	return_sequences=True))
	    model.add(Dropout(0.2))
	    for i in range(1, self.nlayers_lstm-1):
	    	model.add(LSTM(self.nfilters_lstm[i],\
	    		input_shape=(self.sequence_length-1, self.nSequence), \
	    		return_sequences=True))
	    	model.add(Dropout(0.2))
	    model.add(LSTM(self.nfilters_lstm[-1], return_sequences=False))

	    for i in range(self.nlayers_dense):
	    	model.add(Dense(self.nfilters_dense[i], activation='linear'))
	    model.add(Dense(1, activation='linear'))
	    model.compile(loss=loss, optimizer=optimizer)
	    print model.summary()
	    return model

#------------------------------------------------------------------------------

	def make_data(self, sequence, ratio=1.0, length=50):
		train = []
		for i in range(len(sequence)-length):
			train.append(sequence[i:i+length])
		train = np.array(train[:int(ratio*len(train))])
		xtrain = train[:, :-1]
		ytrain = train[:, -1]
		xtrain = np.expand_dims(xtrain, axis=2)
		return xtrain, ytrain

#------------------------------------------------------------------------------

	def fit_data(self, sequence, ratio=1.0, \
					batch_size=100, epochs=100, val_split=0.2, \
					verbose=1):
		self.batch_size = batch_size
		self.epochs = epochs
		self.val_split = val_split

		self.xtrain, self.ytrain = \
						self.make_data(sequence, ratio, self.sequence_length)

		self.print_metadata()
		self.hist = self.model.fit(self.xtrain, self.ytrain, \
						batch_size=self.batch_size, epochs=self.epochs, \
						validation_split=self.val_split, verbose=verbose, \
						shuffle=False)
		if self.save:
			self.model.save(self.savemodelname)
		return self.hist

#------------------------------------------------------------------------------

	def get_prediction(self, plot=True):
		y_pred = self.model.predict(self.xtrain)
		if plot:
			plt.figure(figsize=(15,5))
			plt.plot(self.ytrain,'r', lw=2, label='$\mathtt{Data}$')
			plt.plot(y_pred, '--g', lw=2, label='$\mathtt{Predicted}$')
			plt.legend(loc=1, fontsize=16)
			plt.show()
		return y_pred

#==============================================================================

if __name__ == "__main__":
	x = np.linspace(0, 100, 1000)
	y = np.sin(x)

	obj = lstm_sequence(nSequence=1, sequence_length=11, \
							nlayers_lstm=2, nlayers_dense=2, \
							nfilters_lstm=32, nfilters_dense=32, 
							loss='mean_squared_error', optimizer='adadelta', \
							save=False, savemodelname='model.hdf5')

	hist = obj.fit_data(y, ratio=1.0, batch_size=100, epochs=100, \
						val_split=0.2, verbose=1)

	y_pred = obj.get_prediction(plot=True)

#==============================================================================
