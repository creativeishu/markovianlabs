"""
DESCRIPTION
"""

from train_models import train_models_generator as T

#==============================================================================

# Parameters:
#------------

train_dir='/Users/mohammed/Dropbox/irshad2janu/deeplearning_datasets/\
image_classifiers/catsdogs/train/'
valid_dir='/Users/mohammed/Dropbox/irshad2janu/deeplearning_datasets/\
image_classifiers/catsdogs/validation/'
save_dir='/Users/mohammed/Dropbox/irshad2janu/deeplearning_datasets/\
image_classifiers/catsdogs/savedmodels/'
img_width=224
img_height=224
batch_size=32
nchannels=3
rescale=1./255
shear_range=0.2
zoom_range=0.2
horizontal_flip=True

model='vgg19'
weights=None
loss='categorical_crossentropy'
metrics=['accuracy']
optimizer='adadelta'
nb_epoch=50
verbose=1
save=False
savefilename='mymodel.hdf5'
print_metadata=True

#==============================================================================

ob = T(train_dir, valid_dir, save_dir,\
        img_width, img_height, batch_size, \
        nchannels, rescale, shear_range, \
        zoom_range, horizontal_flip)

hist = ob.train_standard_model(model, weights, \
        loss, metrics, optimizer, nb_epoch, verbose, \
        save, savefilename, print_metadata)