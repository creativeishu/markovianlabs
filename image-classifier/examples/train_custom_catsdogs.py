"""
DESCRIPTION
"""

from train_models import train_models_generator as T

#==============================================================================

# Parameters:
#------------

train_dir='/data/mohammed/data/deeplensing/data160/train/'
valid_dir=None
save_dir='/data/mohammed/data/deeplensing/data160/savedmodels/'
img_width=224
img_height=224
batch_size=32
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
loss='categorical_crossentropy'
metrics=['accuracy']
optimizer='adadelta'
nb_epoch=50
verbose=1
save=False
savefilename='data160_conv%i_dense%i.hdf5'%(nlayers_conv, nlayers_dense)
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
