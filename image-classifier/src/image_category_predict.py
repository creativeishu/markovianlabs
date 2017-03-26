import numpy as np 
import os
import cv2
from sys import exit, argv

from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19

from keras.optimizers import SGD

#==============================================================================

class Imagepredict(object):
	"""
	Some comments
	"""

	def __init__(self, model='vgg19'):

		self.dict = self.get_dict()
		sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

		if model=='vgg19':
			self.model = VGG19(weights='imagenet', include_top=True)
			self.model.compile(optimizer=sgd, loss='categorical_crossentropy')
		elif model=='vgg16':
			self.model = VGG16(weights='imagenet', include_top=True)
			self.model.compile(optimizer=sgd, loss='categorical_crossentropy')
		else:
			print "Only two models are available: VGG16 or VGG19"
			exit()
		print "Class is initialised!!!"


	def get_dict(self):
		path = '/Users/%s/Dropbox/irshad2janu/deeplearning_datasets/image_classifiers/vggfiles/'%os.getlogin()
		class_dict = {}
		index_file = open(path+'synset_words.txt')
		for i,line in enumerate(index_file):
			record = line.rstrip().split(' ')
			class_dict[i] = record[1:]
		return class_dict


	def get_image_input(self, imagepath, shape=(224, 224)):
		im = cv2.resize(cv2.imread(imagepath), shape).astype(np.float32)
		im[:,:,0] -= 103.939
		im[:,:,1] -= 116.779
		im[:,:,2] -= 123.68
		im = np.expand_dims(im, axis=0)
		return im


	def predict_image(self, imagepath, k=5):
		im = self.get_image_input(imagepath)
		out = self.model.predict(im)
		idxs = np.argsort(out[0])[::-1][:k]
		# results = [['class', 'prob']]
		results = []
		for x in idxs:
			 category = [self.dict[x], float(out[0][x])]
			 results.append(category)
		return results

#==============================================================================	
"""
if len(argv)==2:
	input_image = argv[1]
	ob = Imagepredict()
	ob.predict_image(input_image)
else:
	print "Usage: python <script.py> <image_file_path>"
"""
#==============================================================================