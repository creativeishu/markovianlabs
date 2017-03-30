import numpy as np 
import matplotlib.pyplot as plt 
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

		if model=='vgg19':
			self.model = VGG19(weights='imagenet', include_top=False)
		elif model=='vgg16':
			self.model = VGG16(weights='imagenet', include_top=False)
		else:
			print "Only two models are available: VGG16 or VGG19"
			exit()
		print "Class is initialised!!!"


	def get_image_input(self, imagepath, shape=(224, 224)):
		im = cv2.resize(cv2.imread(imagepath), shape).astype(np.float32)
		im[:,:,0] -= 103.939
		im[:,:,1] -= 116.779
		im[:,:,2] -= 123.68
		im = np.expand_dims(im, axis=0)
		return im


	def get_features(self, imagepath, k=5):
		im = self.get_image_input(imagepath)
		out = self.model.predict(im)
		return np.ndarray.flatten(out)

#==============================================================================	

if __name__ == "__main__":
	if len(argv)==2:
		input_image = argv[1]
		ob = Imagepredict()
		out = ob.get_features(input_image)
		print out.shape
		# plt.plot(out)
		# plt.show()

	else:
		print "Usage: python <script.py> <image_file_path>"

#==============================================================================