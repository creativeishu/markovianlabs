import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model, load_model
from keras.layers import Flatten, Dense, Dropout
from keras import backend as K
from keras.optimizers import SGD
from keras.utils import np_utils
from glob import glob

np.random.seed(1234)
from sys import argv, exit

__author__ = 'irshad mohammed'

#==============================================================================

Train_bottleneck = False
Train_topmodel = False

if len(argv)<2 or len(argv)>3:
    print "Usage: python cnn2d_generator.py <train_path> \
    <validation_path (OPTIONAL)>"
    exit()
elif len(argv)==2:
    train_data_dir = argv[1]
    validation_data_dir = None
else:
    train_data_dir = argv[1]
    validation_data_dir = argv[2]

# vgg19, vgg16, inceptionv3, resnet50, xception
modelname = 'vgg16'
img_width, img_height = 224, 224
nb_epoch = 2
batch_size = 32
nchannels = 3
verbose = 1
savemodel = True

loss = 'categorical_crossentropy'
optimizer = SGD(lr=0.0005, momentum=0.9)
metrics = ['accuracy']

#==============================================================================

nTrain = []
class_folders = glob(train_data_dir+'*')
for i in range(len(class_folders)):
    files = glob(class_folders[i]+'/*')
    nTrain.append(len(files))

if validation_data_dir != None:
    nValidation = []
    class_folders = glob(validation_data_dir+'*')
    for i in range(len(class_folders)):
        files = glob(class_folders[i]+'/*')
        nValidation.append(len(files))

# print sum(nTrain), sum(nValidation)

#==============================================================================

DIR = train_data_dir.replace('train', 'savedmodels')
if not os.path.exists(DIR):
    os.mkdir(DIR)
savefilename = DIR + 'finetuned_%s_epoch%i_batch%i.hdf5'\
                        %(modelname, nb_epoch, batch_size)
bottleneck_train_file = DIR+'%s_bottleneck_features_train.npy'%modelname
bottleneck_validation_file = DIR+'%s_bottleneck_features_validation.npy'\
                            %modelname
top_model_file = DIR+'%s_top_model.hdf5'%modelname

#==============================================================================

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, \
    zoom_range=0.2, horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(train_data_dir, \
    target_size=(img_width, img_height), batch_size=batch_size, \
    class_mode='categorical')
train_samples = train_generator.samples 
nb_class = max(train_generator.classes)+1
if train_samples != sum(nTrain):
    print "Something wrong with number of training samples"
    exit()

if validation_data_dir != None:
    test_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = test_datagen.flow_from_directory(\
        validation_data_dir, \
        target_size=(img_width, img_height), batch_size=batch_size, \
        class_mode='categorical')
    valid_samples = validation_generator.samples
    if valid_samples != sum(nValidation):
        print "Something wrong with number of training samples"
        exit()

#==============================================================================

if (K.image_data_format()=='channels_last'):
    inputshape = (img_width, img_height, nchannels)
elif (K.image_data_format()=='channels_first'):
    inputshape = (nchannels, img_width, img_height)
else:
    print "Invalid  data format: ", K.image_data_format()
    exit()

#==============================================================================

if modelname=='vgg19':
    from keras.applications.vgg19 import VGG19
    base_model = VGG19(weights='imagenet', include_top = False, \
                       input_shape=inputshape)
elif modelname=='vgg16':
    from keras.applications.vgg16 import VGG16
    base_model = VGG16(weights='imagenet', include_top = False, \
                       input_shape=inputshape)
elif modelname=='inceptionv3':
    from keras.applications.inception_v3 import InceptionV3
    base_model = InceptionV3(weights='imagenet', include_top = False, \
                       input_shape=inputshape)
elif modelname=='resnet50':
    from keras.applications.resnet50 import ResNet50
    base_model = ResNet50(weights='imagenet', include_top = False, \
                       input_shape=inputshape)
elif modelname=='xception':
    from keras.applications.xception import Xception
    base_model = Xception(weights='imagenet', include_top = False, \
                       input_shape=inputshape)        
else:
    print "Valid models are:"
    print "vgg19, vgg16, inceptionv3, resnet50, xception"
    exit()

#==============================================================================

print "============================================"
print 
print "Summary"
print "-------"
print 
print "Train data directory: ", train_data_dir
if validation_data_dir != None:
    print "Vaoidation data directory: : ", validation_data_dir
print "bottleneck_features will be saved/loaded from directory: ", DIR
print "Top_model will be saved/loaded from directory: ", top_model_file
print "Model will be saved at: ", savefilename
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
print 
print "============================================"

#==============================================================================

def save_bottlebeck_features():
    print "============================================"
    print 
    print "Training bottleneck_features for training set"
    print 

    datagen = ImageDataGenerator(rescale=1. / 255)
    generator = datagen.flow_from_directory(train_data_dir, \
        target_size=(img_width, img_height), batch_size=batch_size, \
        class_mode=None, shuffle=False)
    bottleneck_features_train = base_model.predict_generator(\
                            generator,train_samples/batch_size+1)
    np.save(open(bottleneck_train_file, 'w'), bottleneck_features_train)

    if validation_data_dir != None:
        print "Training bottleneck_features for validation set"
        datagen = ImageDataGenerator(rescale=1. / 255)
        generator = datagen.flow_from_directory(validation_data_dir, \
            target_size=(img_width, img_height), batch_size=batch_size, \
            class_mode=None, shuffle=False)
        bottleneck_features_validation = base_model.predict_generator(\
                            generator, valid_samples/batch_size+1)
        np.save(open(bottleneck_validation_file, 'w'), \
            bottleneck_features_validation)
    print 
    print "============================================"    

#------------------------------------------------------------------------------

def one_hot_encode_object_array(arr):
    '''One hot encode a numpy array of objects (e.g. strings)'''
    uniques, ids = np.unique(arr, return_inverse=True)
    return np_utils.to_categorical(ids, len(uniques))

#------------------------------------------------------------------------------

def train_top_model():
    print "============================================"    
    print
    print "Training top_model..."
    print 
    train_data = np.load(open(bottleneck_train_file))
    train_labels = []
    for i in range(nb_class):
        train_labels +=  list([i] * nTrain[i])
    train_labels = np.array(train_labels)
    train_labels = one_hot_encode_object_array(train_labels)

    if validation_data_dir != None:
        validation_data = np.load(open(bottleneck_validation_file))
        validation_labels = []
        for i in range(nb_class):
            validation_labels +=  list([i] * nValidation[i])
        validation_labels = np.array(validation_labels)
        validation_labels = one_hot_encode_object_array(validation_labels)

    top_model = Sequential()
    top_model.add(Flatten(input_shape=train_data.shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dense(128, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(nb_class, activation='sigmoid'))

    top_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)  

    print top_model.summary()
    if validation_data_dir != None:
        top_model.fit(train_data, train_labels,
              epochs=nb_epoch,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    else:
        top_model.fit(train_data, train_labels,
              epochs=nb_epoch,
              batch_size=batch_size)        
    top_model.save(top_model_file)
    print 
    print "============================================"        
    return top_model

#==============================================================================

if Train_bottleneck:
    save_bottlebeck_features()

if Train_topmodel:    
    top_model = train_top_model()
else:
    print "Loading top model"
    top_model = load_model(top_model_file)

#==============================================================================

print "Joining VGG network to previously training fully connected layer"
print
model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
for layer in model.layers[:15]:
    layer.trainable = False
    
model.compile(loss=loss, optimizer=optimizer, metrics=metrics)  

print "============================================" 
print 
print model.summary()
print
print "============================================" 

#==============================================================================
print "============================================" 
print
print "Finally fine tuning the last block of VGG and \
the fully connected layer..."
print

if validation_data_dir != None:
    model.fit_generator(train_generator, validation_data=validation_generator,\
        steps_per_epoch=train_samples/batch_size+1, epochs=nb_epoch, \
        validation_steps=valid_samples/batch_size+1, verbose=verbose)
else:
    model.fit_generator(train_generator, \
        steps_per_epoch=train_samples/batch_size+1, \
        epochs=nb_epoch, verbose=verbose)
if savemodel:
    model.save(savefilename, overwrite=True)
print 
print "============================================"
#==============================================================================

