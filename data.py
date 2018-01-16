import numpy as np
from config import cnf
import keras

data_cnf = cnf.Coffee()

def load_data(data_set):
	datadir = '/home/zywang/Downloads/CNN_tsc-master/UCR_TS_Archive_2015/'+ data_set + '/' + data_set
	#datadir = '/home/zywang/Downloads/CNN_tsc-master/Elec_10/Elec-5-size/'+ data_set + '/' + data_set
	data_test = np.loadtxt(datadir+'_TEST',delimiter=',')
	data_train = np.loadtxt(datadir+'_TRAIN',delimiter=',')
	X_train = data_train[:,1:]
	y_train = data_train[:,0]

	X_test = data_test[:,1:]
	y_test = data_test[:,0]

	base = np.min(y_train)  #Check if data is 0-based
	
	if base == -1:
	    for i in range(len(y_train)):
	        if y_train[i] == -1:
	        	y_train[i] = 0
	    for i in range(len(y_test)):
	    	if y_test[i] == -1:
	    		y_test[i] = 0

	elif base != 0:
	    y_train -=base
	    y_test -= base

	X_train = X_train.reshape(-1,1,data_cnf.FEATURE_LEN,1)
	y_train = keras.utils.to_categorical(y_train)

	X_test = X_test.reshape(-1,1,data_cnf.FEATURE_LEN,1)
	y_test = keras.utils.to_categorical(y_test)

	return X_train, y_train, X_test, y_test

def get_one_sample(data_set):
	datadir = '/home/zywang/Downloads/CNN_tsc-master/UCR_TS_Archive_2015/'+ data_set + '/' + data_set
	#datadir = '/home/zywang/Downloads/CNN_tsc-master/Elec_10/Elec-5-size/'+ data_set + '/' + data_set
	data_test = np.loadtxt(datadir+'_TEST',delimiter=',')
	X_test = data_test[:,1:]
	test_sample = X_test[1]

	return test_sample