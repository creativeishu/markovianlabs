"""
IBM data stocks
"""

import os
import sys
import csv
import numpy as np 
from sys import exit
import matplotlib.pyplot as plt 
from keras import backend as K
import theano.tensor as T
sys.path.append('/Users/%s/github/markovianlabs/regressor/src'%os.getlogin())

import pickle
from lstm_sequence import lstm_sequence as L

#==============================================================================

__author__ = "Irshad Mohammed"

#==============================================================================
    
def mse_custom(y_true, y_pred):
    x = (y_true[:-1]-y_true[1:])/(y_pred[:-1]-y_pred[1:])
    z = T.switch(T.lt(x,0), 1, 0) 
    return K.mean(z)

#==============================================================================

np.random.seed(5)

max_values = 100000
ratio=1.0
# filename = '/Users/%s/Dropbox/irshad2janu/deeplearning_datasets/regression/\
# sequences/ibm_stocks.csv'%os.getlogin()

filename = 'data/table_twtr_5year.csv'
delimiter=","
col = 5

nSequence=1
sequence_length=21
nlayers_lstm=5
nlayers_dense=2
nfilters_lstm=64
nfilters_dense=32
loss='mse'
optimizer='adadelta'
batch_size=100
epochs=100
val_split=0.4
verbose=1

save=True
savemodelname='models/twtr_seq%i_lstm%i_dense%i_loss%s_optimizer%s_epochs%i.hdf5'\
    %(sequence_length, nlayers_lstm, nlayers_dense, loss, optimizer, epochs)

plot=True
saveplot = True
saveplotfilename = 'figures/twtr.png'

#==============================================================================


with open(filename) as f:
    data = csv.reader(f, delimiter=delimiter)
    power = []
    nb_of_values = 0
    for line in data:
        try:
            power.append(float(line[col]))
            nb_of_values += 1
        except ValueError:
            pass
        if nb_of_values >= max_values:
            break
power = np.array(power)
y = power[::-1]

# plt.plot(y)
# plt.axvline(x=len(y)*(1-val_split))
# plt.show()
# exit()

obj = L(nSequence, sequence_length, nlayers_lstm, nlayers_dense, \
		nfilters_lstm, nfilters_dense, loss, optimizer, save, savemodelname)

hist = obj.fit_data(y, ratio, batch_size, epochs, val_split, verbose)

y_pred = obj.get_prediction(plot=plot, save=saveplot, savefilename=saveplotfilename)

same = 0
diff = 0
for i in range(len(y_pred)-1):
	if (y_pred[i+1]-y_pred[i])/(y[i+1]-y[i]) >= 0.0:
		same += 1
	else:
		diff += 1

print ("success fraction: ", float(same)/(same+diff))

#==============================================================================
