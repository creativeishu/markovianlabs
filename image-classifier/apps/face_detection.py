# Usage: python face_detection.py

import cv2
import copy
import numpy as np 
import face_recognition
from sys import argv, exit
from vggface import VGGFace
from keras.models import Model
import matplotlib.pyplot as plt 
from scipy.spatial.distance import cosine

#==============================================================================

__author__ = "Irshad Mohammed"

#==============================================================================

class FaceDetection(object):
	"""
	Main doc string
	"""
	def __init__(self):
		"""
		Constructor doc string
		"""
		facemodel = VGGFace()
		self.model = Model(inputs=facemodel.layers[0].input, \
								outputs=facemodel.layers[-2].output)

#------------------------------------------------------------------------------

	def get_facesfromimagepath(self, imagePath, \
							img_shape=(224,224), plot=False):
		image = face_recognition.load_image_file(imagePath)
		face_locations = face_recognition.face_locations(image)
		allfaces = []
		for face_location in face_locations:
			top, right, bottom, left = face_location
			face_image = cv2.resize(image[top:bottom, left:right], img_shape)
			allfaces.append(face_image)
			if plot:
				cv2.rectangle(image, (left, top), (right, bottom), \
								(0, 255, 0), 2)
		allfaces = np.array(allfaces)
		if plot:
			print "Total number of faces: ", len(allfaces)
			plt.imshow(image)
			plt.xticks([], [])
			plt.yticks([], [])
			plt.show()
		return allfaces

#------------------------------------------------------------------------------

	def get_featuresfromimagearray(self, image_array):
		im = image_array.astype(np.float32)
		aux = copy.copy(im)
		im[:, :, 0] = aux[:, :, 2]
		im[:, :, 2] = aux[:, :, 0]
		im[:, :, 0] -= 93.5940
		im[:, :, 1] -= 104.7624
		im[:, :, 2] -= 129.1863
		im = np.expand_dims(im, axis=0)
		res = self.model.predict(im)
		res = np.ndarray.flatten(res)
		return res

#------------------------------------------------------------------------------

	def compare_imagearrays(self, imgarr1, imgarr2, plot=False):
		feature1 = self.get_featuresfromimagearray(imgarr1)
		feature2 = self.get_featuresfromimagearray(imgarr2)
		# similarity = 1.0 - cosine(feature1, feature2)
		similarity = 1.0 - np.std(feature1-feature2)
		if plot:
			f, axarr = plt.subplots(1, 3, \
									sharex=False, sharey=False, \
									figsize=(15,5))
			f.subplots_adjust(wspace=0.01,hspace=0.01)
			# similarity = np.exp(-((1-similarity)/0.03)**2)
			axarr[0].imshow(imgarr1)
			axarr[0].set_xticks([], [])
			axarr[0].set_yticks([], [])
			axarr[1].imshow(imgarr2)
			axarr[1].set_xticks([], [])
			axarr[1].set_yticks([], [])
			axarr[2].plot(feature1, feature2, 'x', lw=0.2, \
							label='Similarity = %1.5f'%similarity)
			axarr[2].plot(feature1, feature1, 'k')
			axarr[2].set_xticks([], [])
			axarr[2].set_yticks([], [])
			axarr[2].legend(loc=2, fontsize=14)
			plt.show()
		return similarity

#------------------------------------------------------------------------------

	def compare_images(self, imgpath1, imgpath2, plot=False):
		allfaces1 = self.get_facesfromimagepath(imgpath1)
		allfaces2 = self.get_facesfromimagepath(imgpath2)
		print "Number of faces in 1st image: ", len(allfaces1)
		print "Number of faces in 2nd image: ", len(allfaces2)
		all_similarities = np.zeros((len(allfaces1), len(allfaces2)))
		for i in range(len(allfaces1)):
			for j in range(len(allfaces2)):
				all_similarities[i,j] = self.compare_imagearrays(\
										allfaces1[i], allfaces2[j], \
										plot=plot)
		return all_similarities

#==============================================================================

if __name__ == "__main__":
	if len(argv)==2:
		obj = FaceDetection()
		obj.get_facesfromimagepath(argv[1], plot=True)
	elif len(argv)==3:
		obj = FaceDetection()
		obj.compare_images(argv[1], argv[2], True)
	else:
		print "Usage: "
		print "To identify faces: python face_detection.py <Imgpath>"
		print "To compare faces in two images: \
python face_detection.py <Imgpath1> <Imgpath2>"
		exit()

