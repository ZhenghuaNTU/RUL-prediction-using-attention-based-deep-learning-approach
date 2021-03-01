# -*- coding: utf-8 -*-
"""
Code for FD001 dataset

The first part of the code is for data processing
and the second part is for training and testing

@author: xxx
"""

################ data processing ##########################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from scipy import interpolate

min_max_scaler = preprocessing.MinMaxScaler()

RUL_01 = np.loadtxt('RUL_FD001.txt')
train_01_raw = np.loadtxt('train_FD001.txt')
test_01_raw = np.loadtxt('test_FD001.txt')


train_01_raw[:,2:] = min_max_scaler.fit_transform(train_01_raw[:,2:])
test_01_raw[:,2:] = min_max_scaler.transform(test_01_raw[:,2:])

train_01_nor = train_01_raw
test_01_nor = test_01_raw

train_01_nor = np.delete(train_01_nor, [5,9,10,14,20,22,23], axis=1) # select sensor
test_01_nor = np.delete(test_01_nor, [5,9,10,14,20,22,23], axis=1)  # 

max_RUL = 130.0 #max RUL for training
winSize = 30
trainX = []
trainY = []
testX = []
testY = []

regr = linear_model.LinearRegression() # feature of linear coefficient

def fea_extract(data): # feature extraction of two features
    fea = []
    #print(data.shape)
    x = np.array(range(data.shape[0]))
    for i in range(data.shape[1]):
        fea.append(np.mean(data[:,i]))
        regr.fit(x.reshape(-1,1),np.ravel(data[:,i]))
        fea = fea+list(regr.coef_)
        #print(regr.coef_)
    return fea
    
testInd = []
testLen = []

for i in range(1,int(np.max(train_01_nor[:,0]))+1):
    ind =np.where(train_01_nor[:,0]==i)
    ind = ind[0]
    data_temp = train_01_nor[ind,:]
    for j in range(len(data_temp)-winSize+1):
        trainX.append(data_temp[j:j+winSize,2:].tolist())
        train_RUL = len(data_temp)-winSize-j
        if train_RUL > max_RUL:
            train_RUL = max_RUL
        trainY.append(train_RUL)

for i in range(1,int(np.max(test_01_nor[:,0]))+1):     

    ind =np.where(test_01_nor[:,0]==i)
    ind = ind[0]
    testLen.append(len(ind))
    data_temp = test_01_nor[ind,:]
    
    if len(data_temp)<winSize:
        data_temp_a = []
        for myi in range(data_temp.shape[1]):
            x1 = np.linspace(0, winSize-1, len(data_temp) )
            x_new = np.linspace(0, winSize-1, winSize)
            tck = interpolate.splrep(x1, data_temp[:,myi])
            a = interpolate.splev(x_new, tck)
            data_temp_a.append(a.tolist())
        data_temp_a = np.array(data_temp_a)
        data_temp = data_temp_a.T
        data_temp = data_temp[:,2:]
    else:
        data_temp = data_temp[-winSize:,2:]
    
    data_temp = np.reshape(data_temp,(1,data_temp.shape[0],data_temp.shape[1]))
    if i == 1:
        testX = data_temp
    else:
        testX = np.concatenate((testX,data_temp),axis = 0)
    if RUL_01[i-1] > max_RUL:
        testY.append(max_RUL)
    else:
        testY.append(RUL_01[i-1])
    
trainX = np.array(trainX)
testX = np.array(testX)


trainY = np.array(trainY)/max_RUL # normalize to 0-1 for training


trainX_fea = []
testX_fea = []
for i in range(len(trainX)):
    data_temp = trainX[i]
    trainX_fea.append(fea_extract(data_temp))

for i in range(len(testX)):
    data_temp = testX[i]
    testX_fea.append(fea_extract(data_temp))


scale = preprocessing.StandardScaler().fit(trainX_fea)
trainX_fea = scale.transform(trainX_fea)
testX_fea = scale.transform(testX_fea)


########################### training and testing ######################

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # choose GPU
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Convolution1D, MaxPooling1D, Input, GRU, merge
from keras.layers import multiply
from keras.models import Model
from sklearn import preprocessing
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import keras.callbacks
from keras import optimizers
from keras.layers.wrappers import Bidirectional
from sklearn import preprocessing
import keras.callbacks
from keras.utils.np_utils import to_categorical
from keras.callbacks import LearningRateScheduler
from keras.layers.normalization import BatchNormalization
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
np.random.seed(7)
from keras.layers.wrappers import Bidirectional
from keras import backend as K
import tensorflow as tf
from keras.regularizers import L1L2
from keras.callbacks import ModelCheckpoint
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *
from keras.layers.merge import concatenate
from attention_utils import get_activations, get_data_recurrent
import time


trainData = trainX
testData = testX
trainTarget = trainY
testTarget = testY

INPUT_DIM = trainData.shape[2]
TIME_STEPS = trainData.shape[1]

SINGLE_ATTENTION_VECTOR = False
APPLY_ATTENTION_BEFORE_LSTM = False

def myScore(y_ture,y_pred):
    score = 0
    for i in range(len(y_pred)):
        if y_ture[i] <= y_pred[i]:
            score = score + np.exp(-(y_ture[i]-y_pred[i])/10.0)-1
        else:
            score = score + np.exp((y_ture[i]-y_pred[i])/13.0)-1
    return score

def attention_3d_block(inputs):
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    #output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    output_attention_mul = multiply([inputs, a_probs])
    return output_attention_mul


def model_attention_applied_after_lstm():
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    model_input_fea = Input(shape = (trainX_fea.shape[1],))
    densefea2 = Dense(10,activation = 'relu')(model_input_fea)
    lstm_units = 50
    lstm_out = LSTM(lstm_units, return_sequences=True)(inputs)
    attention_mul = attention_3d_block(lstm_out)
    attention_mul = Flatten()(attention_mul)
    dense_1 = Dense(50, activation='relu')(attention_mul) 
    drop1 = Dropout(0.2)(dense_1) 
    dense_1 = Dense(10, activation='relu')(drop1) 
    mymerge = concatenate([dense_1, densefea2])
    drop2 = Dropout(0.2)(mymerge) 
    output = Dense(1, activation='linear')(drop2)
    model = Model([inputs,model_input_fea],output)
    return model


def model_attention_applied_before_lstm():
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    model_input_fea = Input(shape = (trainX_fea.shape[1],))
    densefea1 = Dense(50,activation = 'relu')(model_input_fea)
    dropfea = Dropout(0.2)(densefea1)
    densefea2 = Dense(10,activation = 'relu')(dropfea)
    attention_mul = attention_3d_block(inputs)
    lstm_units = 50
    lstm_out = LSTM(lstm_units, return_sequences=False)(attention_mul) 
    dense_0 = Dense(50, activation='relu')(lstm_out)
    drop1 = Dropout(0.2)(dense_0)
    dense_1 = Dense(10, activation='relu')(drop1) 
    mymerge = concatenate([dense_1, densefea2])
    drop2 = Dropout(0.2)(mymerge) 
    output = Dense(1, activation='linear')(drop2)
    model = Model([inputs,model_input_fea],output)
    return model
  


class EpochAccuracy(keras.callbacks.Callback):
  def __init__(self, batch_size):
    self.batch_size = batch_size
                
  def on_train_begin(self, logs={}):
    self.val_loss = []
    self.val_score = []
    self.train_loss = []
    self.train_score = []
    self.pred_labels = []
  def on_epoch_end(self, epoch, logs={}):
    yPreds = model.predict([testData,testX_fea])
    yPreds = yPreds*max_RUL
    yPreds = yPreds.ravel()
    val_loss = mean_squared_error(testTarget,yPreds)
    trainPreds = model.predict([trainData,trainX_fea])
    trainPreds = trainPreds*max_RUL
    trainTarget_nor = trainTarget*max_RUL
    self.val_loss.append(val_loss)
    self.train_loss.append(mean_squared_error(trainTarget_nor,trainPreds))
    self.pred_labels.append(yPreds)
    train_score = myScore(trainTarget_nor,trainPreds)
    self.train_score.append(train_score)
    test_score = myScore(testTarget,yPreds)
    self.val_score.append(test_score)
    print('Test RMSE:',np.sqrt(val_loss),' Test score:',test_score)


hidDim = [64]
for i in range(10): #run 10 times
    print('iteration: ',i)
    
    if APPLY_ATTENTION_BEFORE_LSTM:
        model = model_attention_applied_before_lstm()
    else:
        model = model_attention_applied_after_lstm()
    
    model.compile(optimizer='adam',
              loss='mse',
              metrics=['mse'])
    
    batch_size = 100
    nb_epoch = 32
    batch_size1 = 20

    epochaccuracy = EpochAccuracy(batch_size1)
    callbacklist = [epochaccuracy]
    history = model.fit([trainData,trainX_fea], trainTarget,validation_split=0.00, 
              batch_size=batch_size, verbose=2, nb_epoch=nb_epoch,callbacks=callbacklist)

    yPreds = model.predict([testData,testX_fea])
    yPreds = yPreds*max_RUL
    yPreds = yPreds.ravel()
    test_rmse = np.sqrt(mean_squared_error(testTarget,yPreds))
    test_score = myScore(testTarget,yPreds)
    print('lastScore:',test_score,'lastRMSE',test_rmse)


    scio.savemat('results/proposed'+str(i+1)+'_FD001.mat', dict(y_true=testTarget, y_pred=yPreds)) # save results
    K.clear_session()





