# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 10:25:05 2018

@author: Andrija Master
"""

""" LOGISTICKA REGRESIJA """

def Nestrukturni_fun(train_index, test_index, No_class, testsize2):
    
    import warnings
    warnings.filterwarnings('ignore')
    import seaborn as sns
    sns.set()
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from keras.models import Sequential
    from keras.layers import Dense
    from sklearn.model_selection import train_test_split
    import keras
    
    plt.close('all')
    
    def evZ(x):
        return -np.log(1/x-1)
    
    def sigmoid(x):
        return 1/(1+np.exp(-x))

    staze = np.load('staze.npy')
    staze  = staze.astype(int)
    
    skijasi = pd.read_csv(str(staze[0]),index_col='date1')
    output = skijasi.label
    skijasi = skijasi.drop(['label','vreme_pros'],axis=1)

    
    x_train_com, x_test = skijasi.iloc[train_index,:], skijasi.iloc[test_index,:]
    y_train_com, y_test = output.iloc[train_index], output.iloc[test_index] 
    x_train_un, x_train_st, y_train_un, y_train_st = train_test_split(x_train_com, y_train_com, test_size=testsize2, random_state=31)
    
    no_train = x_train_st.shape[0]
    no_test = x_test.shape[0]
    no_train_un = x_train_un.shape[0]
    
    predictions_test = np.zeros([no_test, No_class])
    predictions1_test = np.zeros([no_test, No_class])
    predictions_rand_test = np.zeros([no_test, No_class])
    predictions_nn = np.zeros([no_test, No_class])
    
    Z_train = np.zeros([no_train, No_class])
    Z_test = np.zeros([no_test, No_class])
    Z1_train = np.zeros([no_train, No_class])
    Z1_test = np.zeros([no_test, No_class])
    Z2_train = np.zeros([no_train, No_class])
    Z2_test = np.zeros([no_test, No_class])
    Z2_train_un = np.zeros([no_train,No_class])
    Z3_test = np.zeros([no_test, No_class])
    Z3_train = np.zeros([no_train, No_class])
     
    skorAUC = np.zeros([1,4])
    skorAUC2 = np.zeros([No_class,4])
    
    Y_train = np.zeros([no_train, No_class])
    Y_test = np.zeros([no_test, No_class])
    
    i=0
    for k in staze:
        
        skijasi = pd.read_csv(str(k),index_col='date1')
        output = skijasi.label
        skijasi = skijasi.drop(['label','vreme_pros'],axis=1)
        
        x_train_com, x_test = skijasi.iloc[train_index,:], skijasi.iloc[test_index,:]
        y_train_com, y_test = output.iloc[train_index], output.iloc[test_index] 
        x_train_un, x_train_st, y_train_un, y_train_st = train_test_split(x_train_com, y_train_com, test_size=testsize2, random_state=31)
        
        std_scl = StandardScaler()
        std_scl.fit(x_train_un)
        x_train_un = std_scl.transform(x_train_un)
        x_test = std_scl.transform(x_test)
        x_train_st = std_scl.transform(x_train_st)
        
        Y_train[:,i] = y_train_st.values
        Y_test[:,i] = y_test.values
        
        rand_for = RandomForestClassifier(n_estimators=100)
        rand_for.fit(x_train_un, y_train_un)
        predictions_rand_trainun = rand_for.predict_proba(x_train_un)
        predictions_rand_test[:,i] = rand_for.predict_proba(x_test)[:,1]
        Z3_train[:,i] = evZ(rand_for.predict_proba(x_train_st)[:,1])
        Z3_test[:,i] = evZ(rand_for.predict_proba(x_test)[:,1])
        
        modelOF = Sequential()
        modelOF.add(Dense(30, input_dim = x_train_st.shape[1], activation='relu'))
        modelOF.add(Dense(20, activation='relu'))
        modelOF.add(Dense(14, activation='relu'))
        modelOF.add(Dense(1, activation='sigmoid'))
        modelOF.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'])
        modelOF.fit(x_train_st, y_train_st, epochs=600, batch_size=x_train_st.shape[0],validation_data=(x_train_st, y_train_st))
        
        model2OF = Sequential()
        model2OF.add(Dense(30, input_dim=x_train_st.shape[1], weights = modelOF.layers[0].get_weights() ,activation='relu'))
        model2OF.add(Dense(20,weights = modelOF.layers[1].get_weights() , activation='relu'))
        model2OF.add(Dense(14,weights = modelOF.layers[2].get_weights() , activation='relu'))
        model2OF.add(Dense(1 , weights = modelOF.layers[3].get_weights(), activation='linear'))
        
        model = Sequential()
        model.add(Dense(30, input_dim = x_train_un.shape[1], activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(14, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'])
        ES = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto', baseline=None)
        model.fit(x_train_un, y_train_un, epochs=600, batch_size=250,validation_data=(x_test, y_test), callbacks=[ES])
        
        model2 = Sequential()
        model2.add(Dense(30, input_dim=x_train_un.shape[1], weights = model.layers[0].get_weights() ,activation='relu'))
        model2.add(Dense(20,weights = model.layers[1].get_weights() , activation='relu'))
        model2.add(Dense(14,weights = model.layers[2].get_weights() , activation='relu'))
        model2.add(Dense(1 , weights = model.layers[3].get_weights(), activation='linear'))
        
        logRegression = LogisticRegression(C = 1, penalty = 'l2')
        logRegression1 = LogisticRegression(C = 1, penalty = 'l1', solver='saga')
        logRegression.fit(x_train_un, y_train_un)
        logRegression1.fit(x_train_un, y_train_un)
        
        predictions_train_un = logRegression.predict_proba(x_train_un)
        predictions_test[:,i] = logRegression.predict_proba(x_test)[:,1]
        predictions1_train_un = logRegression1.predict_proba(x_train_un)
        predictions1_test[:,i] = logRegression1.predict_proba(x_test)[:,1]
        
        Z_train[:,i] = logRegression.decision_function(x_train_st)
        Z_test[:,i] = logRegression.decision_function(x_test)
        Z1_train[:,i] = logRegression1.decision_function(x_train_st)
        Z1_test[:,i] = logRegression1.decision_function(x_test)        
        
        Z2_train[:,i] = model2.predict(x_train_st).reshape(no_train)
        Z2_train_un[:,i] = model2OF.predict(x_train_st).reshape(no_train)
#        Z2_train_un[:,i] = evZ(rand_for.predict_proba(x_train_un)[:,1])
        Z2_test[:,i] = model2.predict(x_test).reshape(no_test)
        predictions_nn[:,i] = sigmoid(Z2_test[:,i])
        skorAUC2[i,0] = roc_auc_score(y_test.values,predictions_test[:,i])
        skorAUC2[i,1] = roc_auc_score(y_test.values,predictions1_test[:,i])    
        skorAUC2[i,2] = roc_auc_score(y_test.values,predictions_nn[:,i])
        skorAUC2[i,3] = roc_auc_score(y_test.values,predictions_rand_test[:,i])   
        i+=1
    
    y_test = Y_test
    skorAUC[:,0] = roc_auc_score(y_test.reshape([y_test.shape[0]*y_test.shape[1]]),predictions_test.reshape([y_test.shape[0]*y_test.shape[1]]))
    skorAUC[:,1] = roc_auc_score(y_test.reshape([y_test.shape[0]*y_test.shape[1]]),predictions1_test.reshape([y_test.shape[0]*y_test.shape[1]]))
    skorAUC[:,2] = roc_auc_score(y_test.reshape([y_test.shape[0]*y_test.shape[1]]),predictions_nn.reshape([Z_test.shape[0]*Z2_test.shape[1]]))
    skorAUC[:,3] = roc_auc_score(y_test.reshape([y_test.shape[0]*y_test.shape[1]]),predictions_rand_test.reshape([y_test.shape[0]*y_test.shape[1]]))
    skorAUC2com = np.mean(skorAUC2, axis=0)
    
    Z_train_fin = np.concatenate((Z_train.reshape([Z_train.shape[0]*Z_train.shape[1],1]), \
                        Z1_train.reshape([Z1_train.shape[0]*Z1_train.shape[1],1])),axis=1)
    Z_test_fin = np.concatenate((Z_test.reshape([Z_test.shape[0]*Z_test.shape[1],1]), \
                        Z1_test.reshape([Z1_test.shape[0]*Z1_test.shape[1],1])),axis=1)
    Z_train_fin = np.concatenate((Z_train_fin, Z2_train.reshape([Z2_train.shape[0]*Z2_train.shape[1],1])),axis = 1)
    Z_test_fin = np.concatenate((Z_test_fin, Z2_test.reshape([Z2_test.shape[0]*Z2_test.shape[1],1])), axis = 1)
    Z_train_com = np.concatenate((Z_train_fin, Z3_train.reshape([Z3_train.shape[0]*Z3_train.shape[1],1])),axis = 1)
    Z_test_com = np.concatenate((Z_test_fin, Z3_test.reshape([Z3_test.shape[0]*Z3_test.shape[1],1])), axis = 1)
    
    np.save('Skor_com_AUC.npy', skorAUC)
    np.save('Skor_com_AUC2.npy', skorAUC2)
    np.save('Z_train_com', Z_train_com)
    np.save('Z_test_com.npy', Z_test_com)
    np.save('Z_train_un.npy', Z2_train_un)
    
    Noinst_train = np.round(Z_train_com.shape[0]/No_class).astype(int)
    Noinst_test = np.round(Z_test_com.shape[0]/No_class).astype(int)
    
    Z_train_com[Z_train_com == -np.inf] = -10
    Z_train_com[Z_train_com == -10] = np.min(Z_train_com)-10
    Z_test_com[Z_test_com == -np.inf] = -10
    Z_test_com[Z_test_com == -10] = np.min(Z_test_com)-10
    Z2_train_un[Z2_train_un == -np.inf] = -10
    Z2_train_un[Z2_train_un == -10] = np.min(Z2_train_un)-10
    
    Z_train_com[Z_train_com == np.inf] = 10
    Z_train_com[Z_train_com == 10] = np.max(Z_train_com)+10
    Z_test_com[Z_test_com == np.inf] = 10
    Z_test_com[Z_test_com == 10] = np.max(Z_test_com)+10
    Z2_train_un[Z2_train_un == np.inf] = 10
    Z2_train_un[Z2_train_un == 10] = np.max(Z2_train_un)+10


#    for i in range(Z_train_com.shape[1]):
#        Range = np.abs(np.max(Z_train_com[:,i]) + np.min(Z_train_com[:,i]))
#        faktor = int(math.log10(Range))
#        Z_train_com[:,i] = Z_train_com[:,i]*10**(-faktor)
#        Z_test_com[:,i] = Z_test_com[:,i]*10**(-faktor)
    
    return skorAUC, skorAUC2com, Z_train_com, Z_test_com, Z2_train_un, Noinst_train, Noinst_test, Y_train, Y_test

    