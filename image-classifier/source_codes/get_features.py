"""
Description
"""

import numpy as np
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model

__author__ = "Irsahd Mohammed"

#==============================================================================

def get_features(img_path, layername='fc1'):
	base_model = VGG19(weights='imagenet')
	model = Model(inputs=base_model.input, \
			outputs=base_model.get_layer(layername).output)

	img = image.load_img(img_path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	features = model.predict(x)
	return features

#------------------------------------------------------------------------------

def get_features_vector(img_path, layername='fc1'):
	features = get_features(img_path, layername)
	return np.ndarray.flatten(features)

#==============================================================================

if __name__ == "__main__":
	from sys import argv, exit
	
	if len(argv)==2:
		get_features(argv[1])
	elif len(argv)==3:
		get_features(argv[1], argv[2])
	else:
		print "Usage: python get_features.py <ImagePath> <LayerName (OPTIONAL)>"
		exit()

#==============================================================================		