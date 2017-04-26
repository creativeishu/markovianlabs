"""
IBM data stocks
"""

import os
import sys
import csv
import numpy as np 
from sys import exit
import matplotlib.pyplot as plt 
sys.path.append('/Users/%s/github/markovianlabs/regressor/src'%os.getlogin())

import pickle
from lstm_sequence import lstm_sequence as L

#==============================================================================

__author__ = "Irshad Mohammed"

#==============================================================================

max_values = 100000
filename = '/Users/%s/Dropbox/irshad2janu/deeplearning_datasets/regression/\
sequences/ibm_stocks.csv'%os.getlogin()
with open(filename) as f:
    data = csv.reader(f, delimiter=",")
    power = []
    nb_of_values = 0
    for line in data:
        try:
            power.append(float(line[6]))
            nb_of_values += 1
        except ValueError:
            pass
        if nb_of_values >= max_values:
            break
power = np.array(power)
y = power

obj = L(nSequence=1, sequence_length=11, \
						nlayers_lstm=3, nlayers_dense=2, \
						nfilters_lstm=128, nfilters_dense=32, 
						loss='mean_squared_error', optimizer='adadelta', \
						save=True, savemodelname='ibm.hdf5')

hist = obj.fit_data(y, ratio=1.0, batch_size=100, epochs=500, \
					val_split=0.2, verbose=1)

y_pred = obj.get_prediction(plot=True)

same = 0
diff = 0
for i in range(len(y_pred)-1):
	if (y_pred[i+1]-y_pred[i])/(y[i+1]-y[i]) >= 0.0:
		same += 1
	else:
		diff += 1

print same, diff, same+diff, len(y_pred)-1
print "success fraction: ", float(same)/(same+diff)

#==============================================================================