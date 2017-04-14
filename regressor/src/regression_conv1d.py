import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv1D

np.random.seed(1337)  # for reproducibility

__author__ = "Irshad Mohammed"

#==============================================================================

filename = "/Users/mohammed/Dropbox/fermilabwork/with_myself/\
photoredshifts/data/CMASS_50features_irshad.csv"
nCol = 50
nb_samples_train = 300000
nb_filters = 256

dataframe = pandas.read_csv(filename)
dataset = dataframe.values
X_train = dataset[:nb_samples_train, 1:nCol+1]
Y_train = dataset[:nb_samples_train, nCol+1]
X_test = dataset[nb_samples_train:, 1:nCol+1]
Y_test = dataset[nb_samples_train:, nCol+1]

X_train = X_train.reshape(X_train.shape[0], nCol, 1)
X_test = X_test.reshape(X_test.shape[0], nCol, 1)
input_shape = (nCol, 1)

print "Train data shape: ", X_train.shape, Y_train.shape
print "Valid data shape: ", X_test.shape, Y_test.shape

#==============================================================================

model = Sequential()
model.add(Conv1D(nb_filters, 3, padding='valid', \
                  input_shape=input_shape, activation='relu'))
model.add(Conv1D(nb_filters, 3, activation='relu'))
model.add(Conv1D(nb_filters, 3, activation='relu'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(nb_filters, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_filters, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mean_absolute_percentage_error', optimizer='adadelta')
print model.summary()

model.fit(X_train, Y_train, batch_size=512, epochs=1, verbose=1, validation_data=(X_test, Y_test))
model.save('model.hdf5')

#==============================================================================