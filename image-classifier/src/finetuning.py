
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model, load_model
from keras.layers import ZeroPadding2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.applications.vgg16 import VGG16

np.random.seed(1234)
from sys import argv, exit

#==============================================================================

path = argv[1]
folder = path+'savedmodels/'

train_data_dir = path+'train'
validation_data_dir = path+'validation'

nb_class = 2
img_width, img_height = 150, 150
nb_train_samples = 2222
nb_validation_samples = 1222

# bottleneck training parameters
Train_bottleneck = False
Train_topmodel = False

batch_size = 16
n_epoch = 50

#==============================================================================

if (K.image_data_format()=='channels_last'):
    inputshape = (img_width, img_height, 3)
elif (K.image_data_format()=='channels_first'):
    inputshape = (3, img_width, img_height)
else:
    print "Invalid  data format: ", K.image_data_format()
    exit()

#==============================================================================

def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    np.save(open(folder+'bottleneck_features_train.npy', 'w'),
            bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    np.save(open(folder+'bottleneck_features_validation.npy', 'w'),
            bottleneck_features_validation)

#------------------------------------------------------------------------------

def train_top_model():
    train_data = np.load(open(folder+'bottleneck_features_train.npy'))
    nb_train_samples = len(train_data)
    train_labels = []
    for i in range(nb_class):
        train_labels +=  list([i] * (nb_train_samples / nb_class))

    validation_data = np.load(open(folder+'bottleneck_features_validation.npy'))
    nb_validation_samples = len(validation_data)
    validation_labels = []
    for i in range(nb_class):
        validation_labels +=  list([i] * (nb_validation_samples / nb_class))
    
    train_labels = np.array(train_labels)
    validation_labels = np.array(validation_labels)
    
    print "Training data: ", train_data.shape
    print "Validation data: ", validation_data.shape
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])
    print model.summary()
    model.fit(train_data, train_labels,
              epochs=n_epoch,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save(folder+'top_model.hdf5')
    return model

#==============================================================================

if Train_bottleneck:
    save_bottlebeck_features()

if Train_topmodel:    
    top_model = train_top_model()
else:
    top_model = load_model(folder+'top_model.hdf5')

#==============================================================================

base_model = VGG16(weights='imagenet', include_top=False, input_shape=inputshape)
print base_model.summary()
print
print "VGG model loaded!!!"
print    

#==============================================================================

print "Joining VGG network to previously training fully connected layer"
print
model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
for layer in model.layers[:15]:
    layer.trainable = False
    
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])  
print model.summary()
print

#==============================================================================

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary')

#------------------------------------------------------------------------------

print
print "Finally fine tuning the last block of VGG and the fully connected layer..."
print

model.fit_generator(train_generator, steps_per_epoch=nb_train_samples // batch_size, \
    epochs=n_epoch, validation_data=validation_generator, validation_steps=nb_validation_samples // batch_size)
model.save(folder+'finetunedmodel.hdf5')

#==============================================================================

