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
from sklearn.cluster import KMeans as KM


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
		print 'Shape of all feature array: ', self.all_features.shape

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
		                        figsize=(3*(len(self.resultnames)+1), 3))
		f.subplots_adjust(wspace=0, hspace=0)
		axarr[0].imshow(cv2.resize(cv2.imread(self.imgpath), (224, 224)))
		axarr[0].set_xlabel('$\mathtt{Query}$', fontsize=22)
		axarr[0].set_xticks([], [])
		axarr[0].set_yticks([], [])
		for i in range(1, len(self.resultnames)+1):
			axarr[i].imshow(cv2.resize(cv2.imread(self.resultnames[i-1]), (224, 224)))
			axarr[i].set_xticks([], [])
			axarr[i].set_yticks([], [])
			axarr[i].set_xlabel('$\mathtt{Result\ %i}$'%i, fontsize=22)
		plt.show()

#------------------------------------------------------------------------------

	def get_clustering_labels(self, nClusters=4, plot=False):
		kmeans = KM(n_clusters=nClusters, random_state=0).fit(self.all_features)
		self.clustering_labels = kmeans.labels_
		if plot:
			self.plot_clustering()
		return self.clustering_labels

#------------------------------------------------------------------------------

	def plot_clustering(self, ncol=5):
		f, axarr = plt.subplots(max(self.clustering_labels)+1, ncol, \
		                        sharex=False, sharey=False, \
		                        figsize=(ncol*3, 3*(max(self.clustering_labels)+1)))
		f.subplots_adjust(wspace=0, hspace=0)

		for i in range(max(self.clustering_labels)+1):
			inds = np.where(self.clustering_labels==i)[0]
			for j in range(min(ncol, len(inds))):
				axarr[i,j].imshow(cv2.resize(cv2.imread(self.filenames[inds[j]]), (224,224)))
			axarr[i,0].set_ylabel('$\mathtt{Class\ %i}$'%i, fontsize=22)

		for i in range(max(self.clustering_labels)+1):
			for j in range(ncol):
				axarr[i,j].set_xticks([], [])
				axarr[i,j].set_yticks([], [])

#==============================================================================

if __name__ == "__main__":
	if len(sys.argv)==3:
		obj = VisualSearch(sys.argv[1])
		obj.get_closest_filenames(sys.argv[2], plot=True)
	else:
		print "Usage: python visual_search.py <DIR_name> <FILE_name>"
		sys.exit()


