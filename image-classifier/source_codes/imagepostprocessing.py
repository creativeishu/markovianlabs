"""
a long description
"""

import numpy as np 
import os
import cv2
from sys import exit, argv
import matplotlib.pyplot as plt 

from keras.models import load_model
from keras.models import Model


import numpy as np 
import matplotlib.pyplot as plt 

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
import cv2


__author__ = "Irshad Mohammed"

#==============================================================================

class image_analysis(object):
	"""
	class string
	"""

	def __init__(self, model='vgg19'):
		"""
		doc string constructor
		"""
		firstlayer_index = 0
		if model=='vgg19':
		    from keras.applications.vgg19 import VGG19
		    self.model = VGG19(weights='imagenet', include_top = True)
		elif model=='vgg16':
		    from keras.applications.vgg16 import VGG16
		    self.model = VGG16(weights='imagenet', include_top = True)
		elif model=='inceptionv3':
			from keras.applications.inception_v3 import InceptionV3
			self.model = InceptionV3(weights='imagenet', include_top = True)
		elif model=='resnet50':
		    from keras.applications.resnet50 import ResNet50
		    self.model = ResNet50(weights='imagenet', include_top = True)
		elif model=='xception':
		    from keras.applications.xception import Xception
		    self.model = Xception(weights='imagenet', include_top = True)        
		elif model.endswith('.hdf5'):
			self.model = load_model(model)
			firstlayer_index = 1
		else:
		    print "Valid models are:"
		    print "vgg19, vgg16, inceptionv3, resnet50, xception"
		    print "xception/inceptionv3 model is only available in tf backend"
		    print "Or provide path to a saved model in .hdf5 format"
		    exit()
		self.inputshape = self.model.layers[firstlayer_index].output_shape[1:]

#------------------------------------------------------------------------------

	def get_labels(self, path):
		"""
		my doc string
		"""
		labels = []
		f = open(path, 'r')
		while True:
			line = f.readline()
			if not line:
				break
			labels.append(line)
		f.close()
		return labels

#------------------------------------------------------------------------------

	def get_image_input(self, imagepath):
		"""
		my doc string
		"""		
		shape = self.inputshape[:2]
		im = cv2.resize(cv2.imread(imagepath), shape).astype(np.float32)
		im[:,:,0] -= 103.939
		im[:,:,1] -= 116.779
		im[:,:,2] -= 123.68
		im = np.expand_dims(im, axis=0)
		return im

#------------------------------------------------------------------------------

	def get_image_category(self, imagepath=None, labelpath=None, k=5):
		if imagepath==None or labelpath==None:
			print "Usage: predict_image(imagepage, labelpath)"
			exit()
		im = self.get_image_input(imagepath)
		out = self.model.predict(im)
		idxs = np.argsort(out[0])[::-1][:k]
		labels = self.get_labels(labelpath)
		results = np.array(labels, dtype='str')[idxs]
		probs = np.array(out[0,idxs], dtype='float')
		dictionary = {}
		for i in range(k):
			dictionary[results[i].replace('\n','')] = probs[i]
		return dictionary

#------------------------------------------------------------------------------

	def get_features(self, img_path, layername='fc1'):
		model = Model(inputs=self.model.input, \
				outputs=self.model.get_layer(layername).output)
		im = self.get_image_input(img_path)
		features = model.predict(im)
		return features

#------------------------------------------------------------------------------

	def get_features_vector(self, img_path, layername='fc1'):
		features = self.get_features(img_path, layername)
		return np.ndarray.flatten(features)

#------------------------------------------------------------------------------

	def get_layernames(self):
	    allnames = []
	    for i in range(len(self.model.layers)):
	        names = self.model.layers[i].name
	        allnames.append(names)
	    return allnames		

#------------------------------------------------------------------------------

	def plot_conv_features(self, image, layer=None, \
						save=False, savefilename='features.eps'):
		if layer==None:
			names = self.get_layernames()
			for i in range(len(names)):
				if 'conv' in names[i]:
					layer=names[i]
					break
		elif type(layer)==int:
			names = self.get_layernames()
			layer = names[layer]

		if not (('conv' in layer) or ('pool' in layer)):
			print "please provide name or index of convolution/pooling layer"
			exit()

		features = self.get_features(image, layer)
		N = features.shape[-1]
		nrow = int(N**0.5)
		ncol = int(N**0.5)

		print "================================================="
		print "Layer name: ", layer
		print "Features shape: ", features.shape
		print "Number of rows and columns: ", nrow, ncol
		print "================================================="

		f, axarr = plt.subplots(nrow, ncol, \
	    	sharex=True, sharey=True, figsize=(20,20))
		# f.suptitle('$\mathtt{%s}$'%layer.replace('_', " "), fontsize=22)
		f.subplots_adjust(wspace=0.02, hspace=0.02)
		for i in range(nrow):
		    for j in range(ncol):
		        axarr[i,j].imshow(features[0,:,:,i*ncol+j], cmap='jet')
		        axarr[i,j].set_xticks([], [])
		        axarr[i,j].set_yticks([], [])
		if save:
			f.savefig(savefilename)
		else:
			plt.show()

#------------------------------------------------------------------------------

	def plot_all_conv_features(self, image, DIR='figures/'):
		if not os.path.exists(DIR):
			os.mkdir(DIR)
		names = self.get_layernames()
		for i in range(len(names)):
		    if ('conv' in names[i]) or ('pool' in names[i]):
		    	filename = DIR+names[i]+'.eps'
		        self.plot_conv_features(image, names[i], \
		        	save=True, savefilename=filename)

#==============================================================================
#==============================================================================
#==============================================================================

class TestSetAnalysis(object):
	"""
	class string
	"""

	def __init__(self, model='vgg19'):
		"""
		doc string constructor
		"""
		firstlayer_index = 0
		if model=='vgg19':
		    from keras.applications.vgg19 import VGG19
		    self.model = VGG19(weights='imagenet', include_top = True)
		elif model=='vgg16':
		    from keras.applications.vgg16 import VGG16
		    self.model = VGG16(weights='imagenet', include_top = True)
		elif model=='inceptionv3':
			from keras.applications.inception_v3 import InceptionV3
			self.model = InceptionV3(weights='imagenet', include_top = True)
		elif model=='resnet50':
		    from keras.applications.resnet50 import ResNet50
		    self.model = ResNet50(weights='imagenet', include_top = True)
		elif model=='xception':
		    from keras.applications.xception import Xception
		    self.model = Xception(weights='imagenet', include_top = True)        
		elif model.endswith('.hdf5'):
			self.model = load_model(model)
			firstlayer_index = 1
		else:
		    print "Valid models are:"
		    print "vgg19, vgg16, inceptionv3, resnet50, xception"
		    print "xception/inceptionv3 model is only available in tf backend"
		    print "Or provide path to a saved model in .hdf5 format"
		    exit()
		self.inputshape = self.model.layers[firstlayer_index].output_shape[1:]

#------------------------------------------------------------------------------

	def predict_generator(self, data_dir, batchsize=32):
		datagen = ImageDataGenerator(rescale=1./255)
		generator = datagen.flow_from_directory(data_dir, \
								target_size=self.inputshape[:2], \
		                        batch_size=batchsize, \
		                        class_mode='categorical', \
		                        shuffle=False)

		nfiles = []
		class_folders = glob(data_dir+'*')
		for i in range(len(class_folders)):
		    files = glob(class_folders[i]+'/*')
		    nfiles.append(len(files))

		samples = generator.samples
		nb_class = generator.num_class
		predictions = self.model.predict_generator(generator, \
												samples/batchsize)
		predict_labels = np.argmax(predictions, axis=1)
		true_labels = []
		for i in range(nb_class):
			true_labels +=  list([i] * nfiles[i])
		return true_labels, predict_labels, predictions

#------------------------------------------------------------------------------

	def get_confusion_matrix(self, true_labels, predict_labels):
		return confusion_matrix(true_labels, predict_labels)

	def plot_confusion_matrix(self, true_labels, predict_labels, cmap='Blues'):
		matrix = get_confusion_matrix(true_labels, predict_labels)
		plt.figure(figsize=(8,8))
		plt.imshow(matrix, cmap=cmap)
		plt.show()

#------------------------------------------------------------------------------

	def get_roc_curve(self, true_labels, predictions):
		FPR, TPR, thresholds = roc_curve(true_labels, predictions[:,1])

	def plot_roc_curve(self, true_labels, predictions):
		FPR, TPR, thresholds = get_roc_curve(self, true_labels, predictions)
		plt.figure(figsize=(8,8))
		plt.plot(FPR, TPR, 'k', lw=2)
		plt.xlabel('$\mathtt{FalsePositiveRate}$', fontsize=22)
		plt.ylabel('$\mathtt{TruePositiveRate}$', fontsize=22)
		plt.show()

#==============================================================================

if __name__ == "__main__":
	from sys import argv
	model = 'vgg19'
	ob = image_analysis(model)
	print ob.get_image_category(argv[1], argv[2])

