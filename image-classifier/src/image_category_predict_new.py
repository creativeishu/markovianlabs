import numpy as np 
import os
import cv2
from sys import exit, argv

from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19

# from keras.backend import backend
# from keras.backend import image_dim_ordering as K
# print backend()
# print K()
# exit()

#==============================================================================

class Imagepredict(object):
	"""
	Some comments
	"""

	def __init__(self, model='vgg19'):

		self.dict = self.get_dict()

		if model=='vgg19':
			self.model = VGG19(weights='imagenet', include_top=True)
		elif model=='vgg16':
			self.model = VGG16(weights='imagenet', include_top=True)
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
		im = np.expand_dims(im, axis=0)
		return im


	def predict_image(self, imagepath, k=5):
		im = self.get_image_input(imagepath)
		out = self.model.predict(im)
		idxs = np.argsort(out[0])[::-1][:k]
		for x in idxs:
			print(self.dict[x]), out[0][x]

#==============================================================================	


if len(argv)==2:
	input_image = argv[1]
	ob = Imagepredict()
	ob.predict_image(input_image)
else:
	print "Usage: python <script.py> <image_file_path>"

#==============================================================================