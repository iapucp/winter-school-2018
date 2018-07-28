"""
# Youtube video tutorial: https://www.youtube.com/watch?v=Se9ByBnKb0o&list=PLXO45tsB95cJHXaDKpbwr5fC_CCYylw1f

This code is a modified version of both tutorials:
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
https://github.com/MorvanZhou/tutorials/tree/master/tensorflowTUT
"""
import pandas.core.ops as ops
import csv
import dataSet_ts as dt
#import tensorflow as tf
import numpy as np

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.datasets import mnist
from keras.utils import np_utils
from keras import initializers#initializations
from keras import optimizers

# set random seed for comparing the two result calculations
#tf.set_random_seed(1)

# hyperparameters
lr = 0.001
batch_size = 171 # 2565 / 5
numBatchs = 15
instXday = 3
#batchSizTest = len(XtestSet)

n_inputs = 22   # MNIST data input (img shape: 28*28)
n_steps = 136    # time steps
n_hidden_units = 110   # neurons in hidden layer

# Loading the dataset
paddType = 1
nChannels = 1
reader = dt.ReaderTS(n_steps, n_inputs, instXday, paddType, nChannels)
XdataSet, YdataSet, n_classes, _, arrLens = reader.generateDataSet()
indexRandom =  np.random.permutation(2565)
batchDataSets = np.reshape(indexRandom,(numBatchs,batch_size))
indexAccSet = np.reshape(batchDataSets[0:numBatchs-1],(batch_size*(numBatchs-1)))
X_train = XdataSet[indexAccSet]
Y_train = YdataSet[indexAccSet]
X_test = XdataSet[batchDataSets[-1]]
Y_test = YdataSet[batchDataSets[-1]]

# Build LSTM network
model = Sequential()
model.add(LSTM(n_hidden_units, input_shape=(n_steps,n_inputs)))
model.add(Dense(n_classes, activation='softmax', init=initializers.random_normal(stddev=0.1)))

adamOpt = optimizers.Adam(lr=lr)

model.compile(optimizer=adamOpt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train
num_epochs=1000
history = model.fit(X_train, Y_train, nb_epoch=num_epochs, batch_size=batch_size, shuffle=True, verbose=2)

# Evaluate
evaluation = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=1)
print('Summary: Loss over the test dataset: %.2f, Accuracy: %.2f' % (evaluation[0], evaluation[1]))
