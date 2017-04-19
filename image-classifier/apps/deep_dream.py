'''
Deep dream implementation in keras.

Usage:
python deep_dream.py <path_to_the_image>
'''


from __future__ import print_function

import numpy as np
from scipy.misc import imsave
from scipy.optimize import fmin_l_bfgs_b

from keras.preprocessing.image import load_img, img_to_array
from keras import backend as K
from keras.layers import Input

from keras.applications import vgg19
from keras.applications import vgg16
from keras.applications import inception_v3
from keras.applications import resnet50

import sys


__author__ = 'jverma'





class DeepDream:
	"""
	Implements deep dreaming in Keras.
	"""
	def __init__(self, model='vgg19', img_height=600, img_width=600):
		"""
		Instantiates the class.

		Parameters
		----------
		model: vgg19, vgg16, inceptionv3, resnet50.
		img_height: Default is 600.
		img_width: Default is 600.
		"""
		self.img_height = img_height
		self.img_width = img_width
		self.img_size = (img_height, img_width, 3)

		self.dream_layer = Input(batch_shape=(1,) + self.img_size)
		
		if model=='vgg19':
			self.base_model = vgg19.VGG19(input_tensor=self.dream_layer, weights='imagenet', include_top = False)
		elif model=='vgg16':
			self.base_model = vgg16.VGG16(input_tensor=self.dream_layer, weights='imagenet', include_top = False)
		elif model=='inceptionv3':
			self.base_model = inception_v3.InceptionV3(input_tensor=self.dream_layer, weights='imagenet', include_top = False)
		elif model=='resnet50':
			self.base_model = resnet50.ResNet50(input_tensor=self.dream_layer, weights='imagenet', include_top = False)      
		else:
			print("Valid models are:")
			print("vgg19, vgg16, inceptionv3, resnet50")
			exit()
		print("Model loaded...")


	def preprocess_image(self, img_path):
		"""
		Util function to open, resize and format images into tensors.

		Paramaters
		----------
		img_path: Path to the input image.

		Returns
		-------
		A numpy array encoding the processed image.
		"""
		img = load_img(img_path, target_size=(self.img_height, self.img_width))
		img = img_to_array(img)
		img = np.expand_dims(img, axis=0)
		img[:,:,0] -= 103.939
		img[:,:,1] -= 116.779
		img[:,:,2] -= 123.68
		return img


	def deprocess_image(self, img_arr):
		"""
		Util function to convert the tensor into a valid image.

		Parameters
		----------
		img_arr: Input tensor encoding the image.

		Returns
		-------
		RGB for the image.
		"""
		img_arr = img_arr.reshape(self.img_size)
		img_arr[:,:,0] += 103.939
		img_arr[:,:,1] += 116.779
		img_arr[:,:,2] += 123.68

		# BGR --> RGB
		img_arr = img_arr[:, :, ::-1]
		img_arr = np.clip(img_arr, 0, 255).astype('uint8')
		return img_arr


	def compute_loss(self, settings):
		"""
		Computes the loss function according to the defined settings.

		Parameters
		----------
		settings: A nested dictionary containing the settings to use. e.g. for vgg
		settings = {'features': {'block5_conv1': 0.05,
                            'block5_conv2': 0.02},
               'continuity': 0.1,
               'dream_l2': 0.02,
               'jitter': 0}


        Returns
        -------
        The loss function.
		"""
		layer_dict = dict([(layer.name, layer) for layer in self.base_model.layers])
		loss = K.variable(0.)
		for layer_name in settings['features']:
			assert layer_name in layer_dict, 'Layer ' + layer_name + ' not found in the model.'
			coeff = settings['features'][layer_name]
			x = layer_dict[layer_name].output
			x_shape = layer_dict[layer_name].output_shape

			# avoid artifacts by only involving non-border pixels.
			loss -= coeff * K.sum(K.square(x[:, 2:x_shape[1]-2, 2:x_shape[2]-2, :]))

			# add continuity loss (gives local coherence and can result in artful blur)
			a = K.square(self.dream_layer[:, :self.img_height-1, :self.img_width-1, :] - self.dream_layer[:, 1:, :self.img_width-1, :])
			b = K.square(self.dream_layer[:, :self.img_height-1, :self.img_width-1, :] - self.dream_layer[:, :self.img_height-1, 1:, :])
			continuity_loss = K.sum(K.pow(a+b, 1.25))

			loss += settings['continuity'] * continuity_loss / np.prod(self.img_size)

			# add image L2 norm loss (prevants pixels from taking very high values, makes image darker)
			loss += settings['dream_l2'] * K.sum(K.square(self.dream_layer)) / np.prod(self.img_size)

			return loss


	def compute_grads(self, loss):
		"""
		Computes the gradients of the loss.

		Parameters
		----------
		loss: the loss function.

		Returns
		-------
		The gradient function.
		"""
		grads = K.gradients(loss, self.dream_layer)

		outputs = [loss]
		if isinstance(grads, (list, tuple)):
			outputs += grads
		else:
			outputs.append(grads)
		f_outputs = K.function([self.dream_layer], outputs) 
		return f_outputs



#######################################################################
		

class Evaluator(object):
    """Loss and gradients evaluator.
    This Evaluator class makes it possible
    to compute loss and gradients in one pass
    while retrieving them via two separate functions,
    "loss" and "grads". This is done because scipy.optimize
    requires separate functions for loss and gradients,
    but computing them separately would be inefficient.
    """

    def __init__(self, f_outputs, img_size):
    	"""
    	Instantiates the class.

    	Parameters
    	----------
    	f_outputs: The gradient function.
    	img_size: Size of the image.
    	"""
        self.loss_value = None
        self.grad_values = None
        self.f_outputs = f_outputs
        self.img_size = img_size


    def loss(self, x):
    	"""
    	Computes the value of the loss function at a point.

    	Parameters
    	----------
    	x: Input tensor.

    	Returns
    	-------
    	Value of the loss.
    	"""
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x, self.f_outputs, self.img_size)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
    	"""
    	Computes the value of the gradient function at a point.

    	Parameters
    	----------
    	x: Input tensor.

    	Returns
    	-------
    	Value of the gradient.
    	"""
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values



##########################################################################

def eval_loss_and_grads(x, f_outputs, img_size):
	"""
	Util function to evaluate the loss and the gradients.

	Parameters
    ----------
    x: Input tensor.
    f_outputs: The gradient function.
	img_size: Size of the image.	
	"""
	x = x.reshape((1,) + img_size)
	outs = f_outputs([x])
	loss_value = outs[0]
	if (len(outs[1:]) == 1):
		grad_values = outs[1].flatten().astype('float64')
	else:
		grad_values = np.array(outs[1:]).flatten().astype('float64')
	return loss_value, grad_values


#########################################################################
#########################################################################



if __name__ == "__main__":
	image_path = sys.argv[1]
	
	deep_dream = DeepDream()
	img_size = deep_dream.img_size

	settings = {'features': {'block5_conv1': 0.05,
                            'block5_conv2': 0.02},
               'continuity': 0.1,
               'dream_l2': 0.02,
               'jitter': 0}

	loss = deep_dream.compute_loss(settings)
	grads = deep_dream.compute_grads(loss)


	evaluator = Evaluator(f_outputs=grads, img_size=img_size)


	x = deep_dream.preprocess_image(image_path)

	for i in range(10):
		random_jitter = (settings['jitter'] * 2) * (np.random.random(img_size) - 0.5)
		x += random_jitter

		x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
	                                     fprime=evaluator.grads, maxfun=7)
		print(min_val)

		x = x.reshape(img_size)
		x -= random_jitter
		img = deep_dream.deprocess_image(np.copy(x))
		result_prefix = 'apple'
		fname = result_prefix + '_at_iteration_%d.png' % i
		imsave(fname, img)
		print('Image saved as', fname)











