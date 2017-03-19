import os
import h5py
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import optimizers
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from sys import argv, exit


# dimensions of our images.
img_width, img_height = 150, 150

folder = argv[1]
train_data_dir = folder+'train'
validation_data_dir = folder+'validation'

nb_train_samples = 2222
nb_validation_samples = 1222
nb_epoch = 50 # originally 50

#Train_Bottleneck = False
nb_class = 2

simplemodel = Sequential()
simplemodel.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3)))
simplemodel.add(Activation('relu'))
simplemodel.add(MaxPooling2D(pool_size=(2, 2)))

simplemodel.add(Conv2D(32, (3, 3)))
simplemodel.add(Activation('relu'))
simplemodel.add(MaxPooling2D(pool_size=(2, 2)))

simplemodel.add(Conv2D(64, (3, 3)))
simplemodel.add(Activation('relu'))
simplemodel.add(MaxPooling2D(pool_size=(2, 2)))

simplemodel.add(Flatten())
simplemodel.add(Dense(64))
simplemodel.add(Activation('relu'))
simplemodel.add(Dropout(0.5))
simplemodel.add(Dense(1))
simplemodel.add(Activation('sigmoid'))

simplemodel.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print "Model loaded and compiled"

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary')

print simplemodel.summary()
# simplemodel.fit_generator(
#         train_generator,
#         samples_per_epoch=nb_train_samples,
#         nb_epoch=nb_epoch,
#         validation_data=validation_generator,
#         nb_val_samples=nb_validation_samples)

simplemodel.fit_generator(generator=train_generator, validation_data=validation_generator, \
    steps_per_epoch=35, epochs=50, validation_steps=10)
