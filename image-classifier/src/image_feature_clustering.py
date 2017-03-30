"""
Organize your images into a fixed number of groups.
Next: Find a way to compute number of clusters.
"""

import os
import h5py
import cv2
import sys
import numpy as np 
import json

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans
import subprocess
from collections import defaultdict

import matplotlib.pyplot as plt 

from get_imagenet_features import Imagepredict


__author__ = "jverma"


class_model = Imagepredict()



def build_feature_dict(input_dir):
	"""
	Obtains features for the images in the directory.

	Parameters
	----------
	input_dir: Path to the directory containing the images.

	Returns
	-------
	A dictionary containing the file names and corresponding features.
	"""
	extracted_features = {}
	photos = os.listdir(input_dir)
	for file_name in photos:
		try:
			input_image = os.path.join(input_dir, file_name)
			print input_image
			imagenet_features = class_model.get_features(input_image)
			extracted_features[file_name] = imagenet_features
		except:
			pass
	return extracted_features


def dimensional_reduction(input_array):
	"""
	"""
	dim_red = PCA(n_components=100)
	return dim_red.fit_transform(input_array)







def cluster_features(input_dir, dim_red=True, k=10):
	"""
	Clusters the images in the directory based on thei categories.

	Parameters
	----------
	input_dir: Path to the directory containing the images.
	k: Number of clusters.

	Returns
	-------
	A dictionary containing the cluster indices and files in the clusters.
	"""
	classification_results = build_feature_dict(input_dir)
	image_names = classification_results.keys()
	image_features = np.array(classification_results.values())
	if dim_red:
		X = dimensional_reduction(image_features)
	else:
		X = image_features

	# vectorizer = CountVectorizer(min_df=1)
	# X = vectorizer.fit_transform(image_classes)

	km = KMeans(init='k-means++', n_clusters=k, n_init=10)
	km.fit(X)

	image_cluster_label_dict = defaultdict(list)
	for i,t in enumerate(km.labels_):
		image_cluster_label_dict[t].append(image_names[i])
	return image_cluster_label_dict




def organize_photos(input_dir, dim_red=True, k=10):
	"""
	Groups the images into different folders.

	Parameters
	----------
	input_dir: Path to the directory containing the images.
	k: Number of clusters.
	"""
	image_cluster_labels = cluster_features(input_dir, dim_red=dim_red, k=k)
	for idx in image_cluster_labels:
		dir_name = 'cluster' + str(idx)
		subprocess.call(['mkdir', dir_name])
		for x in image_cluster_labels[idx]:
			file_path = input_dir + '/' + str(x).decode('utf-8')
			subprocess.call(['cp', file_path, dir_name])
	return None



input_dir = sys.argv[1]
organize_photos(input_dir)
# print input_dir

# feature_array = np.array(build_feature_dict(input_dir).values())
# print feature_array.shape
# print "features extracted..."

# dim_red = PCA(n_components=100)
# print "PCA initialized.."
# X = dim_red.fit_transform(feature_array)
# print X.shape
# plt.scatter(X[:,0], X[:,1])
# plt.show()


