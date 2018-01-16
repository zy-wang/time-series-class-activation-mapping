from keras.models import Sequential
from keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense, Activation, GlobalAveragePooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
import numpy as np
from keras.layers.core import Lambda
from keras import backend as K 
import h5py
from config import cnf
from keras import optimizers


data_cnf = cnf.Coffee()

def TSC_convolutions():
	model = Sequential()
	model.add(Conv2D(data_cnf.num_filt_1, (3,1), padding = 'same', input_shape = (1, data_cnf.FEATURE_LEN, 1),name = 'conv1_1'))
	model.add(BatchNormalization(name = 'bn1'))
	model.add(Activation('relu',name = 'rl1') )
	#model.add(MaxPooling2D((1,2), name  = 'mp1'))

	model.add(Conv2D(data_cnf.num_filt_2, (3,1), padding = 'same', name = 'conv2_1'))
	model.add(BatchNormalization(name = 'bn2'))
	model.add(Activation('relu', name = 'rl2'))
	#model.add(MaxPooling2D((1,2),name = 'mp2'))

	model.add(Conv2D(data_cnf.num_filt_3, (3,1), padding = 'same', name = 'conv3_1'))
	model.add(BatchNormalization(name = 'bn3'))
	model.add(Activation('relu', name = 'rl3'))
	#model.add(MaxPooling2D((1,2), name = 'mp3'))
	return model

def get_model():
	model = TSC_convolutions()

	model = load_model_weights(model, '/home/zywang/python/Coffee-4-tsc-cnn.h5')

	model.add(GlobalAveragePooling2D(name = 'gap1') )

	model.add(Dense(data_cnf.NUM_CLASSES, activation= 'softmax', init = 'uniform', name = 'Dense_new'))

	adam = optimizers.Adam(lr = data_cnf.LEARNING_RATE)
	model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])

	return model

def load_model_weights(model, weights_path):
	print ('loading model...')
	
	#model.load_weights(weights_path, by_name=True) #load the weights directly, different from the code in the github, which load weights through h5 file(use the file operation)
	for k in range(len(model.layers)):
		model.layers[k].trainable = True # we needn't the convolution layers trained agian. we just want to train the GlobaMaxPooling layer and Dense layer
	
	print('model loaded.')

	return model

def get_output_layer(model, layer_name):
	#get the symbolic outputs of each 'key' layer (we gave them unqiue names).
	layer_dict = dict([(layer.name, layer) for layer in model.layers])
	layer = layer_dict[layer_name]
	return layer

