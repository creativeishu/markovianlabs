# "usage: python facedetection.py <image_path> <Cascade_file_path>"

import numpy as np 
import cv2
from sys import argv, exit
import matplotlib.pyplot as plt 
import os 
import copy
from vggface import VGGFace
from scipy.spatial.distance import cosine, correlation
import face_recognition
from keras.models import Model


#==============================================================================

base_model = VGGFace()
layername = 'fc6'
model = Model(inputs=base_model.input, \
				outputs=base_model.get_layer(layername).output)

#==============================================================================

imagePath1 = argv[1]
imagePath2 = argv[2]

#==============================================================================


def facedetect(imagePath):
	image = face_recognition.load_image_file(imagePath)
	face_locations = face_recognition.face_locations(image)

	allfaces = []
	for face_location in face_locations:
		top, right, bottom, left = face_location
		face_image = image[top:bottom, left:right]
		face_image = cv2.resize(face_image, (224, 224))
		allfaces.append(face_image)
	allfaces = np.array(allfaces)
	return allfaces

#------------------------------------------------------------------------------

def faceidentify(image, model):
	im = image.astype(np.float32)
	aux = copy.copy(im)
	im[:, :, 0] = aux[:, :, 2]
	im[:, :, 2] = aux[:, :, 0]
	im[:, :, 0] -= 93.5940
	im[:, :, 1] -= 104.7624
	im[:, :, 2] -= 129.1863
	im = np.expand_dims(im, axis=0)
	res = model.predict(im)
	res = np.ndarray.flatten(res)
	return res

#------------------------------------------------------------------------------

def facedetectandidentify(imagePath, model):
	allfaces = facedetect(imagePath)
	data = []
	for i in range(len(allfaces)):
		res = faceidentify(allfaces[i], model)
		data.append([allfaces[i], res])
	return data

#==============================================================================

data1 = facedetectandidentify(imagePath1, model)
data2 = facedetectandidentify(imagePath2, model)

nfaces1 = len(data1)
nfaces2 = len(data2)
print "Found %i faces in image 1"%nfaces1
print "Found %i faces in image 2"%nfaces2

for i in range(nfaces1):
	for j in range(nfaces2):
		f, axarr = plt.subplots(1, 3, sharex=False, sharey=False, figsize=(15,5))
		f.subplots_adjust(wspace=0.01,hspace=0.01)
		similarity = 1-cosine(data1[i][1], data2[j][1])
		similarity = np.exp(-((1-similarity)/0.03)**2)
		axarr[0].imshow(data1[i][0])
		axarr[0].xaxis.set_major_formatter(plt.NullFormatter())
		axarr[0].yaxis.set_major_formatter(plt.NullFormatter())
		axarr[1].imshow(data2[j][0])
		axarr[1].xaxis.set_major_formatter(plt.NullFormatter())
		axarr[1].yaxis.set_major_formatter(plt.NullFormatter())		
		axarr[2].plot(data1[i][1], data2[j][1], '.', lw=0.2, label='Similarity = %1.5f'%similarity)
		axarr[2].plot(data1[i][1], data1[i][1], 'k')
		axarr[2].xaxis.set_major_formatter(plt.NullFormatter())
		axarr[2].yaxis.set_major_formatter(plt.NullFormatter())
		axarr[2].legend(loc=2, fontsize=14)
		plt.show()

#==============================================================================