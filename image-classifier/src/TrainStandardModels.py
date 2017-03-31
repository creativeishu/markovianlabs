import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout
from keras import backend as K
from sys import exit, argv


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

modelname = 'vgg16'
img_width, img_height = 224, 224
nb_epoch = 30
batch_size = 16
nchannels = 3
verbose = 1
savemodel = True

loss = 'binary_crossentropy'
optimizer = 'adadelta'
metrics = ['accuracy']

DIR = train_data_dir.replace('train', 'savedmodels')
if not os.path.exists(DIR):
    os.mkdir(DIR)
savefilename = DIR + '%s_epoch%i_batch%i.hdf5'\
                        %(modelname, nb_epoch, batch_size)

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

def load_standard_models_for_training(model='vgg19', num_classes=2, \
                         inputshape=(224,224,3), weights=None):
    """
    Arguements:
        model: vgg19, vgg16, inceptionv3, resnet50, xception
        num_classes: Number of classes 
        inputshape: (img_width, img_height, n_channels)
        weights: imagenet or None
        
    Returns: 
        A model, neural network architecture. 
    """
    
    if model=='vgg19':
        from keras.applications.vgg19 import VGG19
        base_model = VGG19(weights=weights, include_top = False, \
                           input_shape=inputshape)
    elif model=='vgg16':
        from keras.applications.vgg16 import VGG16
        base_model = VGG16(weights=weights, include_top = False, \
                           input_shape=inputshape)
    elif model=='inceptionv3':
        from keras.applications.inception_v3 import InceptionV3
        base_model = InceptionV3(weights=weights, include_top = False, \
                           input_shape=inputshape)
    elif model=='resnet50':
        from keras.applications.resnet50 import ResNet50
        base_model = ResNet50(weights=weights, include_top = False, \
                           input_shape=inputshape)
    elif model=='xception':
        from keras.applications.xception import Xception
        base_model = Xception(weights=weights, include_top = False, \
                           input_shape=inputshape)        
    else:
        print "Valid models are:"
        print "vgg19, vgg16, inceptionv3, resnet50, xception"
        exit()

    x = Flatten()(base_model.output)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation = 'softmax')(x)
    model = Model(inputs = base_model.input, outputs = predictions)
    return model

model = load_standard_models_for_training(modelname, num_classes=nb_class, \
                                          inputshape=inputshape)
model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

#==============================================================================

print "============================================"
print "Using model: ", modelname
print "Data Format: ", K.image_data_format()
print "Input shape: ", inputshape
print "Number of training samples: ", train_samples
if validation_data_dir != None:
	print "Number of validation samples: ", valid_samples
print "Number of classes: ", nb_class
print "Batch size: ", batch_size
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
