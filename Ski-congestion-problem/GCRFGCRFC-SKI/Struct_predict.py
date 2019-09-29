# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 08:42:01 2019

@author: Andrija Master
"""

import time
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import keras
from keras.layers import Dense
from sklearn.metrics import r2_score
from sgcrf import SparseGaussianCRF
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential


def Strukturni_predict_fun(train_index, test_index, ModelSTNo):
    
    atribute = pd.read_csv('atribute')
    output = pd.read_csv('output')
    atribute = atribute.ix[:,1:].values
    output = output.ix[:,1:].values
    
    timeST = np.zeros([ModelSTNo])
    R2 = np.zeros([ModelSTNo])
    
    x_train, x_test = atribute[train_index,:], atribute[test_index,:]
    y_train, y_test = output[train_index,:], output[test_index,:]
    
    std_scl = StandardScaler()
    std_scl.fit(x_train)
    
    x_train = std_scl.transform(x_train)
    x_test = std_scl.transform(x_test)
    
    model = SparseGaussianCRF()
    
    start_time = time.time()
    model.fit(x_train, y_train)
    y_SGCRF = model.predict(x_test).reshape(-1)
    timeST[0] = time.time() - start_time
    
    
    start_time = time.time()
    model2 = Sequential()
    model2.add(Dense(30, input_dim = x_train.shape[1], activation='relu'))
    model2.add(Dense(25, activation='relu'))
    model2.add(Dense(20, activation='relu'))
    model2.add(Dense(7, activation='linear'))
    model2.compile(loss='mean_absolute_error', optimizer='SGD')
    ES = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=0, mode='auto', baseline=None)
    model2.fit(x_train, y_train, epochs=1200, batch_size=200,validation_data=(x_test, y_test), callbacks=[ES])
    y_NN1 = model2.predict(x_test).reshape(-1)
    timeST[1] = time.time() - start_time 

    start_time = time.time()    
    model3 = Sequential()
    model3.add(Dense(35, input_dim = x_train.shape[1], activation='relu'))
    model3.add(Dense(26, activation='relu'))
    model3.add(Dense(22, activation='relu'))
    model3.add(Dense(7, activation='linear'))
    model3.compile(loss='mean_absolute_error', optimizer='SGD')
    ES = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=0, mode='auto', baseline=None)
    model2.fit(x_train, y_train, epochs=1200, batch_size=200,validation_data=(x_test, y_test), callbacks=[ES])
    y_NN2 = model3.predict(x_test).reshape(-1)
    timeST[2] = time.time() - start_time
    
    start_time = time.time()    
    model4 = Sequential()
    model4.add(Dense(36, input_dim = x_train.shape[1], activation='relu'))
    model4.add(Dense(27, activation='relu'))
    model4.add(Dense(21, activation='relu'))
    model4.add(Dense(7, activation='linear'))
    model4.compile(loss='mean_absolute_error', optimizer='SGD')
    ES = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=0, mode='auto', baseline=None)
    model4.fit(x_train, y_train, epochs=1200, batch_size=200,validation_data=(x_test, y_test), callbacks=[ES])
    y_NN3 = model4.predict(x_test).reshape(-1)
    timeST[3] = time.time() - start_time
    
    R2[0] = r2_score(y_test.reshape(-1), y_SGCRF)
    R2[1] = r2_score(y_test.reshape(-1), y_NN1)
    R2[2] = r2_score(y_test.reshape(-1), y_NN2)
    R2[3] = r2_score(y_test.reshape(-1), y_NN3)

    return timeST, R2
    