"""
DESCRIPTION
"""
import sys
sys.path.append('/Users/mohammed/github/markovianlabs/image-classifier/source_codes/')

import pickle
from train_models import train_models_generator as T

#==============================================================================

# Parameters:
#------------

train_dir='/Users/mohammed/Desktop/lensdata/train/'
valid_dir=None
save_dir='/Users/mohammed/Desktop/lensdata/savedmodels/'
img_width=64
img_height=64
batch_size=20
nchannels=3
rescale=1./255
shear_range=0.2
zoom_range=0.2
horizontal_flip=True

nlayers_conv=5
filters_conv=32
nlayers_dense=2
filters_dense=32
conv_kernel=(3,3)
pooling_kernel=(2,2)
activation='relu'
loss='binary_crossentropy'
metrics=['accuracy']
optimizer='adadelta'
nb_epoch=250
verbose=1
save=True
savefilename='cleandata_convlayers%i_denselayers%i_nepoch%i.hdf5'\
				%(nlayers_conv, nlayers_dense, nb_epoch)
print_metadata=True

#==============================================================================

ob = T(train_dir, valid_dir, save_dir,\
        img_width, img_height, batch_size, \
        nchannels, rescale, shear_range, \
        zoom_range, horizontal_flip)

hist = ob.train_custom_model(nlayers_conv, filters_conv, \
        nlayers_dense, filters_dense, \
        conv_kernel, pooling_kernel, activation, \
        loss, metrics, optimizer, nb_epoch, verbose, \
        save, savefilename, print_metadata)

pickle.dump(hist.history, open(savefilename.replace('.hdf5', '_hist.p'), "w"))

#==============================================================================