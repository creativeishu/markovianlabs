# "usage: python facedetection.py <image_path> <Cascade_file_path>"

import numpy as np 
import cv2
from sys import argv, exit
import matplotlib.pyplot as plt 
import os 
import copy
from vggface import VGGFace
from scipy.spatial.distance import cosine, correlation

#==============================================================================

model = VGGFace()

#==============================================================================


cascPath = '/Users/%s/Dropbox/irshad2janu/deeplearning_datasets/image_classifiers/vggfiles/haarcascade_frontalface_default.xml'%os.getlogin()
imagePath1 = argv[1]
if len(argv)==3:
	imagePath2 = argv[2]

#==============================================================================

def makeimage(imagepath, convertGrey):
	image = cv2.imread(imagepath)
	if convertGrey:
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	return image

#------------------------------------------------------------------------------	

def facedetect(imagePath, cascPath, grey=False, scaleFactor=1.1, minNeighbors=5, minSize=(30,30)):
	image = makeimage(imagePath, grey)
	faceCascade = cv2.CascadeClassifier(cascPath)
	faces = faceCascade.detectMultiScale(image, scaleFactor=scaleFactor, \
		minNeighbors=minNeighbors, minSize=minSize, flags = cv2.CASCADE_SCALE_IMAGE)

	allfaces = []
	for (x, y, w, h) in faces:
		sub_face = image[y:y+h, x:x+w]
		sub_face = cv2.resize(sub_face, (224, 224))
		allfaces.append(sub_face)
		cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
	allfaces = np.array(allfaces)
	return image, allfaces

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
	res = np.reshape(res, (2622))
	return res

#------------------------------------------------------------------------------

def facedetectandidentify(imagePath, cascPath, model):
	image, allfaces = facedetect(imagePath, cascPath)
	data = []
	for i in range(len(allfaces)):
		res = faceidentify(allfaces[i], model)
		data.append([allfaces[i], res])
	return data

#==============================================================================

data1 = facedetectandidentify(imagePath1, cascPath, model)
data2 = facedetectandidentify(imagePath2, cascPath, model)

nfaces1 = len(data1)
nfaces2 = len(data2)
print "Found %i faces in image 1"%nfaces1
print "Found %i faces in image 2"%nfaces2

for i in range(nfaces1):
	for j in range(nfaces2):
		f, axarr = plt.subplots(1, 3, sharex=False, sharey=False, figsize=(15,5))
		f.subplots_adjust(wspace=0.01,hspace=0.01)
		similarity = 1-cosine(data1[i][1], data2[j][1])
		similarity = np.exp(-((1-similarity)/0.1)**2)
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