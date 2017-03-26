import os
import h5py
import cv2
import sys
import numpy as np 
import json

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import subprocess
from collections import defaultdict

from image_category_predict import Imagepredict


__author__ = "jverma"


class_model = Imagepredict()



def classify_photos(input_dir):
	"""
	"""
	category_predictions = {}
	# vocab = []
	photos = os.listdir(input_dir)
	for file_name in photos:
		try:
			input_image = os.path.join(input_dir, file_name)
			predicted_categories = class_model.predict_image(input_image)
			category_list = [' '.join(x[0]) for x in predicted_categories]
			# vocab.extend(category_list)
			category_sen = ' '.join(category_list)
			category_predictions[file_name] = category_sen
		except:
			pass
	# print len(set(vocab))
	return category_predictions




def cluster_photos(input_dir, k=20):
	"""
	"""
	classification_results = classify_photos(input_dir)
	image_names = classification_results.keys()
	image_classes = classification_results.values()

	vectorizer = CountVectorizer(min_df=1)
	X = vectorizer.fit_transform(image_classes)

	km = KMeans(init='k-means++', n_clusters=k, n_init=10)
	km.fit(X)

	image_cluster_label_dict = defaultdict(list)
	for i,t in enumerate(km.labels_):
		image_cluster_label_dict[t].append(image_names[i])
	return image_cluster_label_dict




def organize_photos(input_dir, k=10):
	"""
	"""
	image_cluster_labels = cluster_photos(input_dir, k=k)
	for idx in image_cluster_labels:
		dir_name = str(idx)
		subprocess.call(['mkdir', dir_name])
		for x in image_cluster_labels[idx]:
			file_path = input_dir + '/' + str(x).decode('utf-8')
			subprocess.call(['cp', file_path, dir_name])
	return None



input_dir = sys.argv[1]
organize_photos(input_dir)


