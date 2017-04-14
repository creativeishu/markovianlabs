import os
import sys
sys.path.append('/Users/%s/github/markovianlabs/regressor/src'%os.getlogin())

import pickle
from regression import regression as R

__author__ = "Irshad Mohammed"

#==============================================================================

filename = '/Users/mohammed/Dropbox/fermilabwork/with_myself/\
photoredshifts/data/CMASS_50features_irshad.csv'

shuffle = True

xcols = tuple(range(1, 51))
ycol = 51 
nb_train_samples = 300000
nb_valid_samples = 300000

ncov = 5
ndense = 3
nfilters_cov = 128 
nfilters_dense=128
kernel_size=2
activation_cov='relu'
activation_dense='relu'
loss='mean_absolute_percentage_error'
optimizer='adadelta'
batch_size=1000
nb_epochs=100
verbose=1
save=True
savefilename='model_regression.hdf5'

#==============================================================================

obj = R(filename, shuffle)
obj.make_data(xcols, ycol, nb_train_samples, nb_valid_samples)
hist = obj.train_network(ncov, ndense, \
					nfilters_cov, nfilters_dense, \
					kernel_size, \
					activation_cov, activation_dense, \
					loss, optimizer, \
					batch_size, nb_epochs, verbose, \
					save, savefilename)

pickle.dump( hist.history, open(savefilename.replace('.hdf5', '_hist.p'), "w"))

#==============================================================================