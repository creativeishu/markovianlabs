# A sample CNN network

import os
import h5py
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.convolutional import ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import optimizers
from sys import argv, exit
from keras import backend as K

__author__ = 'irshad mohammed'

#==============================================================================

if len(argv)<2 or len(argv)>3:
	print "Usage: python cnn2d_generator.py <train_path> <validation_path (OPTIONAL)>"
	exit()
elif len(argv)==2:
	train_data_dir = argv[1]
	validation_data_dir = None
else:
	train_data_dir = argv[1]
	validation_data_dir = argv[2]


img_width, img_height = 64, 64
nb_epoch = 50
batch_size = 512
nfilters = 64
nlayers = 3
nchannels = 3
verbose = 1
savemodel = True

loss = 'binary_crossentropy'
optimizer = 'adadelta'
metrics = ['accuracy']

DIR = train_data_dir.replace('train', 'savedmodels')
if not os.path.exists(DIR):
    os.mkdir(DIR)
savefilename = DIR + 'layers%i_filters%i_epoch%i_batch%i.hdf5'\
                        %(nlayers, nfilters, nb_epoch, batch_size)

#==============================================================================

if (K.image_data_format()=='channels_last'):
	inputshape = (img_width, img_height, nchannels)
elif (K.image_data_format()=='channels_first'):
	inputshape = (nchannels, img_width, img_height)
else:
	print "Invalid  data format: ", K.image_data_format()
	exit()

#==============================================================================

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, \
	zoom_range=0.2, horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(train_data_dir, \
	target_size=(img_width, img_height), batch_size=batch_size, \
	class_mode='categorical')
train_samples = train_generator.samples 
nb_class = max(train_generator.classes)+1


if validation_data_dir != None:
	test_datagen = ImageDataGenerator(rescale=1./255)
	validation_generator = test_datagen.flow_from_directory(validation_data_dir, \
		target_size=(img_width, img_height), batch_size=batch_size, \
		class_mode='categorical')
	valid_samples = validation_generator.samples

#==============================================================================

model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=inputshape))
for i in range(nlayers):
	model.add(Conv2D(nfilters, (3, 3), activation='relu', name='conv2d_%i'%(i+1)))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(nfilters, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_class, activation='sigmoid'))

model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

#==============================================================================

print "============================================"
print "Data Format: ", K.image_data_format()
print "Input shape: ", inputshape
print "Number of training samples: ", train_samples
if validation_data_dir != None:
	print "Number of validation samples: ", valid_samples
print "Number of classes: ", nb_class
print "Number of convolutional layers: ", nlayers
print "Batch size: ", batch_size
print "Number of fileters in convolutional layers: ", nfilters
print "Loss: ", loss
print "Optimizer: ", optimizer
print "Metrics: ", metrics
print "Number of epochs: ", nb_epoch
print "Model will be saved at: ", savefilename
print "============================================"
print model.summary()

#==============================================================================

if validation_data_dir != None:
	model.fit_generator(train_generator, validation_data=validation_generator, \
		steps_per_epoch=train_samples//batch_size, epochs=nb_epoch, \
		validation_steps=valid_samples/batch_size, verbose=verbose)
else:
	model.fit_generator(train_generator, \
		steps_per_epoch=train_samples//batch_size, epochs=nb_epoch, verbose=verbose)	
if savemodel:
	model.save(savefilename, overwrite=True)

#==============================================================================
