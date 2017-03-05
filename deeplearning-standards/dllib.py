from keras.layers import Dense, Convolution2D, MaxPooling2D, Activation
from keras.models import Sequential

class Deeplearninglibrary(object):
	def __init__(self):
		print "Deep Learning library is initialised"

	def denselayer(self, nlayers=1, nfilters=[32], input_dim=3, init='normal', activation='relu', nclass=1):
		model = Sequential()
		model.add(Dense(nfilters[0], input_dim=input_dim, init=init, activation=activation))
		for i in range(1, nlayers):
			model.add(Dense(nfilters[i], init=init, activation=activation))
		model.add(Dense(nclass, init=init))	
		return model


#==============================

if __name__=="__main__":
	ob = Deeplearninglibrary()
	model = ob.denselayer(10, [32]*10)
