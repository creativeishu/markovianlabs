import numpy as np 
import matplotlib.pyplot as plt 
from keras.datasets import cifar100
from keras.models import load_model

(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
x_train = x_train/255.0
x_test = x_test/255.0
xtrain = x_train[:,:,:,:2]
ytrain = x_train[:,:,:,2]
xtest = x_test[:,:,:,:2]
ytest = x_test[:,:,:,2]

ytrain = np.reshape(ytrain, \
	(ytrain.shape[0], ytrain.shape[1], ytrain.shape[2], 1))
ytest = np.reshape(ytest, \
	(ytest.shape[0], ytest.shape[1], ytest.shape[2], 1))

print "Training set: ", xtrain.shape, ytrain.shape
print "Test set: ", xtest.shape, ytest.shape

model = load_model('encoder.hdf5')
x = model.predict(xtest[5:6])
plt.imshow(ytest[0,:,:,0]-x[0,:,:,0])
plt.colorbar()
plt.show()
