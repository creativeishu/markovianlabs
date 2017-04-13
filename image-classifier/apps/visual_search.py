import os
import sys

path = '/Users/%s/github/markovianlabs/image-classifier/\
source_codes/'%os.getlogin()
sys.path.append(path)

import numpy as np 
import matplotlib.pyplot as plt 
from imagepostprocessing import image_analysis as IA
from glob import glob
from scipy.spatial.distance import cosine 
import cv2

__author__ = "Irshad Mohammed"


#==============================================================================

class VisualSearch(object):
	"""
	Main doc string
	"""
	def __init__(self, data_dir, modelname='vgg19'):
		"""
		doc string constructor
		"""
		self.data_dir = data_dir
		self.obj_features = IA(modelname)
		print self.obj_features.model.summary()
		self.get_all_features()

#------------------------------------------------------------------------------

	def get_all_features(self):
		self.filenames = glob(self.data_dir+'*')
		self.filenames = np.array(self.filenames, dtype=str)
		print "Target directory: ", self.data_dir
		self.all_features = []
		for i in range(len(self.filenames)):
			print "Featuring image %i of %i"%(i+1, len(self.filenames))
			self.all_features.append(\
				self.obj_features.get_features_vector(self.filenames[i]))
		self.all_features = np.array(self.all_features)
		print self.all_features.shape, 'Shape of all feature array'

#------------------------------------------------------------------------------

	def get_closest_filenames(self, imgpath, k=3, plot=False):
		self.imgpath = imgpath
		my_feature = self.obj_features.get_features_vector(self.imgpath)
		dist = []
		for i in range(len(self.all_features)):
			dist.append(cosine(my_feature, self.all_features[i]))
		ind = np.argsort(dist)[:k]
		self.resultnames = self.filenames[ind]
		if plot:
			self.plot_closest_images()
		return self.resultnames

#------------------------------------------------------------------------------

	def plot_closest_images(self):
		f, axarr = plt.subplots(1, len(self.resultnames)+1, \
		                        sharex=False, sharey=False, \
		                        figsize=(20, 20/(len(self.resultnames)+1)))
		f.subplots_adjust(wspace=0, hspace=0)
		axarr[0].imshow(cv2.imread(self.imgpath))
		axarr[0].set_xticks([], [])
		axarr[0].set_yticks([], [])
		for i in range(1, len(self.resultnames)+1):
		    axarr[i].imshow(cv2.imread(self.resultnames[i-1]))
		    axarr[i].set_xticks([], [])
		    axarr[i].set_yticks([], [])
		plt.show()

#==============================================================================

if __name__ == "__main__":
	if len(sys.argv)==3:
		obj = VisualSearch(sys.argv[1])
		obj.get_closest_filenames(sys.argv[2], plot=True)
	else:
		print "Usage: python visual_search.py <DIR_name> <FILE_name>"
		sys.exit()


