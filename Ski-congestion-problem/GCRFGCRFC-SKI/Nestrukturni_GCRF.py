# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 11:47:32 2018

@author: Andrija Master
"""

def Nestrukturni_fun_GCRF(train_index, test_index, Y_test1, No_class, testsize2):
    
    import warnings
    warnings.filterwarnings('ignore')
    import seaborn as sns
    sns.set()
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import RidgeCV
    from sklearn.linear_model import LassoCV
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    from keras.models import Sequential
    from keras.layers import Dense
    from sklearn.model_selection import train_test_split
    import keras
    
    plt.close('all')
    
    
    staze = np.load('staze.npy')
    staze  = staze.astype(int)
    
    skijasi = pd.read_csv(str(staze[0]),index_col='date1')
    output = skijasi.vreme_pros
    skijasi = skijasi.drop(['label','vreme_pros'],axis=1)
    
    x_train_com, x_test = skijasi.iloc[train_index,:], skijasi.iloc[test_index,:]
    y_train_com, y_test = output.iloc[train_index], output.iloc[test_index] 
    x_train_un, x_train_st, y_train_un, y_train_st = train_test_split(x_train_com, y_train_com, test_size=testsize2, random_state=31)
    
    no_train = x_train_st.shape[0]
    no_test = x_test.shape[0]
    
    Z_train = np.zeros([no_train, No_class])
    Z_test = np.zeros([no_test, No_class])
    Z1_train = np.zeros([no_train, No_class])
    Z1_test = np.zeros([no_test, No_class])
    Z2_train = np.zeros([no_train, No_class])
    Z2_test = np.zeros([no_test, No_class])
    Z2_train_un = np.zeros([no_train,No_class])
    Z3_test = np.zeros([no_test, No_class])
    Z3_train = np.zeros([no_train, No_class])
     
    skorR2 = np.zeros([1,4])
    skorR22 = np.zeros([No_class,4])
    skorMSE = np.zeros([No_class,4])
    
    Y_train = np.zeros([no_train, No_class])
    Y_test = np.zeros([no_test, No_class])
    
    i=0
    for k in staze:
        
        skijasi = pd.read_csv(str(k),index_col='date1')
        output = skijasi.vreme_pros
        skijasi = skijasi.drop(['label','vreme_pros'],axis=1)
                
        x_train_com, x_test = skijasi.iloc[train_index,:], skijasi.iloc[test_index,:]
        y_train_com, y_test = output.iloc[train_index], output.iloc[test_index] 
        x_train_un, x_train_st, y_train_un, y_train_st = train_test_split(x_train_com, y_train_com, test_size=testsize2, random_state=31)
        
        y_test1 = Y_test1[:,i]
        x_train_un = x_train_un[y_train_un!=0]
        x_train_st = x_train_st[y_train_st!=0]
        x_test = x_test[y_test1!=0]           
        y_train_un = y_train_un[y_train_un!=0]
        
        std_scl = StandardScaler()
        std_scl.fit(x_train_un)
        x_train_un = std_scl.transform(x_train_un)
        x_test = std_scl.transform(x_test)
        x_train_st = std_scl.transform(x_train_st)
        
        Y_train[:,i] = y_train_st.values
        Y_test[:,i] = y_test.values
        
        rand_for = RandomForestRegressor(n_estimators=100)
        rand_for.fit(x_train_un, y_train_un)
        Z3_train[:,i][y_train_st.values!=0] = rand_for.predict(x_train_st)
        Z3_test[:,i][y_test1!=0]  = rand_for.predict(x_test)
        
        
        model = Sequential()
        model.add(Dense(10, input_dim = x_train_un.shape[1], activation='relu'))
        model.add(Dense(5, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mean_absolute_error', optimizer='SGD')
        ES = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=0, mode='auto', baseline=None)
        model.fit(x_train_un, y_train_un, epochs=1200, batch_size=200,validation_data=(x_test, y_test1[y_test1!=0]), callbacks=[ES])
        
        linRegression = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5)
        linRegression1 = LassoCV(alphas=[0.1, 1.0, 10.0], cv=5)
        linRegression.fit(x_train_un, y_train_un)
        linRegression1.fit(x_train_un, y_train_un)

        Z_train[:,i][y_train_st.values!=0] = linRegression.predict(x_train_st)
        Z_test[:,i][y_test1!=0] = linRegression.predict(x_test)
        Z1_train[:,i][y_train_st.values!=0] = linRegression1.predict(x_train_st)
        Z1_test[:,i][y_test1!=0] = linRegression1.predict(x_test)        
        
        Z2_train[:,i][y_train_st.values!=0] = model.predict(x_train_st).reshape(x_train_st.shape[0])
        Z2_train_un[:,i] = Z2_train[:,i]
        Z2_test[:,i][y_test1!=0] = model.predict(x_test).reshape(x_test.shape[0])
        skorR22[i,0] = r2_score(y_test.values, Z_test[:,i])
        skorR22[i,1] = r2_score(y_test.values, Z1_test[:,i])    
        skorR22[i,2] = r2_score(y_test.values, Z2_test[:,i])
        skorR22[i,3] = r2_score(y_test.values, Z3_test[:,i])   
        i+=1
    
    y_test = Y_test
    skorR2[:,0] = r2_score(y_test.reshape([y_test.shape[0]*y_test.shape[1]]), Z_test.reshape([y_test.shape[0]*y_test.shape[1]]))
    skorR2[:,1] = r2_score(y_test.reshape([y_test.shape[0]*y_test.shape[1]]), Z1_test.reshape([y_test.shape[0]*y_test.shape[1]]))
    skorR2[:,2] = r2_score(y_test.reshape([y_test.shape[0]*y_test.shape[1]]), Z2_test.reshape([Z2_test.shape[0]*Z2_test.shape[1]]))
    skorR2[:,3] = r2_score(y_test.reshape([y_test.shape[0]*y_test.shape[1]]), Z3_test.reshape([y_test.shape[0]*y_test.shape[1]]))
    skorMSE[:,0] = mean_squared_error(y_test.reshape([y_test.shape[0]*y_test.shape[1]]), Z_test.reshape([y_test.shape[0]*y_test.shape[1]]))
    skorMSE[:,1] = mean_squared_error(y_test.reshape([y_test.shape[0]*y_test.shape[1]]), Z1_test.reshape([y_test.shape[0]*y_test.shape[1]]))
    skorMSE[:,2] = mean_squared_error(y_test.reshape([y_test.shape[0]*y_test.shape[1]]), Z2_test.reshape([Z2_test.shape[0]*Z2_test.shape[1]]))
    skorMSE[:,3] = mean_squared_error(y_test.reshape([y_test.shape[0]*y_test.shape[1]]), Z3_test.reshape([y_test.shape[0]*y_test.shape[1]]))
    skorR22com = np.mean(skorR22, axis=0)
    
    Z_train_fin = np.concatenate((Z_train.reshape([Z_train.shape[0]*Z_train.shape[1],1]), \
                        Z1_train.reshape([Z1_train.shape[0]*Z1_train.shape[1],1])),axis=1)
    Z_test_fin = np.concatenate((Z_test.reshape([Z_test.shape[0]*Z_test.shape[1],1]), \
                        Z1_test.reshape([Z1_test.shape[0]*Z1_test.shape[1],1])),axis=1)
    Z_train_fin = np.concatenate((Z_train_fin, Z2_train.reshape([Z2_train.shape[0]*Z2_train.shape[1],1])),axis = 1)
    Z_test_fin = np.concatenate((Z_test_fin, Z2_test.reshape([Z2_test.shape[0]*Z2_test.shape[1],1])), axis = 1)
    Z_train_com = np.concatenate((Z_train_fin, Z3_train.reshape([Z3_train.shape[0]*Z3_train.shape[1],1])),axis = 1)
    Z_test_com = np.concatenate((Z_test_fin, Z3_test.reshape([Z3_test.shape[0]*Z3_test.shape[1],1])), axis = 1)
    
    np.save('Skor_com_R2.npy', skorR2)
    np.save('Skor_com_R22.npy', skorR22)
    np.save('Skor_com_MSE.npy', skorMSE)
    np.save('Z_train_com', Z_train_com)
    np.save('Z_test_com.npy', Z_test_com)
    np.save('Z_train_un.npy', Z2_train_un)
    
    Noinst_train = np.round(Z_train_com.shape[0]/No_class).astype(int)
    Noinst_test = np.round(Z_test_com.shape[0]/No_class).astype(int)
    
    return np.mean(skorMSE,axis=0), skorR2, skorR22com, Z_train_com, Z_test_com, Z2_train_un, Noinst_train, Noinst_test, Y_train, Y_test