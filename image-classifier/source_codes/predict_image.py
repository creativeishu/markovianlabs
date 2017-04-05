"""
a long description
"""

import numpy as np 
import os
import cv2
from sys import exit, argv

from keras.optimizers import SGD
from keras.models import load_model
from keras import backend as K

__author__ = "Irshad Mohammed"

#==============================================================================

class predict_image(object):
	"""
	class string
	"""

	def __init__(self, model='vgg19'):
		"""
		doc string constructor
		"""
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
		else:
		    print "Valid models are:"
		    print "vgg19, vgg16, inceptionv3, resnet50, xception"
		    print "xception/inceptionv3 model is only available in tf backend"
		    print "Or provide path to a saved model in .hdf5 format"
		    exit()
		self.inputshape = self.model.layers[0].output_shape[1:]
		self.labels = self.get_labels()
		print "Class is initialised!!!"

#------------------------------------------------------------------------------

	def get_labels(self):
		"""
		my doc string
		"""
		path='/Users/mohammed/Dropbox/irshad2janu/\
deeplearning_datasets/image_classifiers/vggfiles/labels'
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

	def predict_image(self, imagepath, k=5):
		im = self.get_image_input(imagepath)
		out = self.model.predict(im)
		idxs = np.argsort(out[0])[::-1][:k]
		results = np.array(self.labels, dtype='str')[idxs]
		return results

#------------------------------------------------------------------------------

	def get_features(self):
		print self.model.output_shape
		self.model.layers.pop()
		print self.model.output_shape


#==============================================================================

if __name__ == "__main__":
	from sys import argv
	model = '/Users/mohammed/Desktop/layers3_filters64_epoch50_batch512.hdf5'
	ob = predict_image('vgg19')
	# print ob.predict_image(argv[1])
	ob.get_features()






