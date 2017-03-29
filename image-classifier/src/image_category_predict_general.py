import numpy as np 
import os
import cv2
from sys import exit, argv

from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.optimizers import SGD
from keras.models import load_model

#==============================================================================

class Imagepredict(object):
	"""
	Some comments
	"""

	def __init__(self, modelpath=None, labelspath=None):
		if modelpath==None or labelspath==None:
			print "Two parameters required:"
			print "modelpath, labelspath"
			exit()
		else:
			self.model = load_model(modelpath)
			self.labels = self.get_labels(labelspath)
		print self.model.summary()
		print "Class is initialised!!!"


	def get_labels(self, path):
		labels = []
		f = open(path, 'r')
		while True:
			line = f.readline()
			if not line:
				break
			labels.append(line)
		f.close()
		return labels


	def get_image_input(self, imagepath, shape=(150, 150)):
		im = cv2.resize(cv2.imread(imagepath), shape).astype(np.float32)
		im /= 255.0
		im = np.expand_dims(im, axis=0)
		return im


	def predict_image(self, imagepath, k=5):
		im = self.get_image_input(imagepath)
		out = self.model.predict(im)
		idxs = np.argsort(out[0])[::-1][:k]
		results = np.array(self.labels, dtype='str')[idxs]
		return results

#==============================================================================	

if __name__ == "__main__":
	modelpath = argv[1]
	labelpath = argv[2]
	input_image = argv[3]
	ob = Imagepredict(modelpath, labelpath)
	print ob.predict_image(input_image)

#==============================================================================