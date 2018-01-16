from keras.models import *
from keras.callbacks import *
import keras.backend as K 
from model import *
from data import *
from config import cnf
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 

data_cnf = cnf.Coffee()

def train(dataset_path):
	model = get_model()
	X_train, y_train, X_test, y_test = load_data(dataset_path)                                                   
	print('training...')
	checkpoint_path = "/home/zywang/python/cam-Coffee-model-para.h5" #the path of model parameters saved file after each epoch
	checkpoint = ModelCheckpoint(checkpoint_path, monitor = 'val_acc', verbose = 0, save_best_only = True, save_weights_only = False, mode = 'max')
	model.fit(X_train,y_train, epochs = 3000, batch_size =data_cnf.BATCH_SIZE, verbose = 1, validation_data = (X_test, y_test), callbacks = [checkpoint])
	return model 



def visualize_class_activation_map(model_path, sample_path, output_path):
	model = load_model(model_path) # load the trained model
	sample = get_one_sample(sample_path)
	#width, height = sample.shape()
	#reshape to the input shape (rows, clos, channels)
	sample = sample.reshape(-1,1,data_cnf.FEATURE_LEN,1) #get one sample

	#get the weights of the globaAveragepool to num_classes
	class_weights = model.layers[-1].get_weights()[0]
	final_conv_layer = get_output_layer(model, 'conv3_1')
	get_output = K.function([model.layers[0].input, K.learning_phase()], [final_conv_layer.output, model.layers[-1].output])#K.learning_phase() is necessary K.learning_phase() = 0 means testing model, =1 means training model
	#get_output = K.function([model.layers[0].input], [final_conv_layer.output]) 
	print('already....')
	[conv_outputs, predictions] = get_output([sample,0])
	conv_outputs = conv_outputs[0,:,:,:]

	#creat the class activation map.
	cam = np.zeros(dtype = np.float32, shape = conv_outputs.shape[0:2])

	for i, w in enumerate(class_weights[:,0]):
		#if w<0:
			#plt.plot(list(conv_outputs[:, :, i][0]))
		
		cam += w*conv_outputs[:,:,i]

	print('predictions----->', predictions)
	cam /= np.max(cam)
	
	#heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
	#cv2.imwrite(output_path, heatmap)
	return cam

if __name__ == '__main__':
	#model1 = train('Coffee')
	cam = visualize_class_activation_map('/home/zywang/python/cam-Coffee-model-para.h5', 'Coffee', '/home/zywang/python/heatmap.jpg')


#