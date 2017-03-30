"""
Visual query search engine.
"""

import os
import h5py
import cv2
import sys
import numpy as np 
import json


from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

import subprocess
from collections import defaultdict


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




def dimensional_reduction(input_array, n_components=100):
	"""
	Performs dimensional reduction on the features. Similar to the LSA.

	Parameters
	----------
	input_array: A numpy array of the feature vectors.
	n_components: Number of components to be projected onto.

	Returns
	-------
	Reduced array.
	"""
	dim_red = PCA(n_components=n_components)
	return dim_red.fit_transform(input_array)







def search_images(query, input_dir, dim_red=False, k=10):
	"""
	Seacrhes the images in the directory for the query image.

	Parameters
	----------
	query: Path to the input query image.
	input_dir: Path to the directory containing the images.
	dim_red: If True, performs dimensional reduction using PCA.
	k: Number of results to be retrieved.

	Returns
	-------
	A list containing the names of the files relevant to the query.
	"""
	classification_results = build_feature_dict(input_dir)
	image_names = classification_results.keys()
	image_features = np.array(classification_results.values())
	if dim_red:
		X = dimensional_reduction(image_features)
	else:
		X = image_features

	query_features = class_model.get_features(query)

	ranking = cosine_similarity(X, query_features)
	image_ids = np.argsort(ranking, axis=0)[::-1]
	ranked_imgs = [image_names[image_ids[i]] for i in range(k)]
	return ranked_imgs


	




def search_results(query, input_dir, dim_red=True, k=10):
	"""
	Extracts the images relevant to the query folder.

	Parameters
	----------
	query: Path to the input query image.
	input_dir: Path to the directory containing the images.
	dim_red: If True, performs dimensional reduction using PCA.
	k: Number of images to be retrieved.

	Returns
	-------
	Saves the relevant images into a folder search_results.
	"""
	image_labels = search_images(query, input_dir, dim_red=False, k=k)
	dir_name = 'search_results'
	subprocess.call(['mkdir', dir_name])

	for x in image_labels:
		file_path = input_dir + '/' + str(x).decode('utf-8')
		subprocess.call(['cp', file_path, dir_name])
	return None





input_dir = sys.argv[1]
query = sys.argv[2]
search_results(query, input_dir)



