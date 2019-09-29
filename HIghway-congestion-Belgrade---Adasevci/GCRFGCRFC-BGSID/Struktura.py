# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 09:14:13 2018

@author: Andrija Master
"""

import pandas as pd
import scipy.stats as sp
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mutual_info_score


def Struktura_fun(No_class,NoGraph,R2,y_train_com, Noinst_train, Noinst_test, koef1 = 0.5):

    Se = np.zeros([NoGraph,No_class,No_class])
    y_train_com = y_train_com.values
    #domenske_veze = pd.read_csv('domen.csv')
    #domenske_veze.set_index('Bolesti', inplace=True)
    ##domenske_veze.set_index('ICD-9-CM CODE', inplace=True)
    #bol = output.columns[1:].values
    #for i in range(bol.shape[0]):
    #    bol[i] = bol[i].split('_')[1]
    #matrica = np.zeros([50,1])
    #df_veza = pd.DataFrame(matrica, index=bol)   
    ##for i in range(domenske_veze.shape[0]):
    ##    domenske_veze.iloc[i,0] = domenske_veze.iloc[i,0].split("'")[1]
    ##    domenske_veze.loc[i,'Hijerarhija1'] = domenske_veze.loc[i,'Hijerarhija1'].split("'")[1]
    ##    domenske_veze.loc[i,'Hijerarhija2'] = domenske_veze.loc[i,'Hijerarhija2'].split("'")[1]
    ##    domenske_veze.loc[i,'Hijerarhija3'] = domenske_veze.loc[i,'Hijerarhija3'].split("'")[1]
    ##domenske_veze = domenske_veze.loc[:,['Bolesti','Hijerarhija1','Hijerarhija2','Hijerarhija3']]
    ##domenske_veze.to_csv
    #for i in range(bol.shape[0]):
    #    provera = domenske_veze[domenske_veze.index==bol[i]]
    #    if len(domenske_veze[domenske_veze.index==bol[i]]) != 0:
    #        df_veza.iloc[i,0] = domenske_veze[domenke_veze.index==bol[i]].values[0,0]  
    #s
    #df_veza.reset_index(inplace=True,drop=True)
    #df_veza = df_veza[df_veza!=0]
    #df_veza.dropna(inplace=True)
    #for i in range(df_veza.shape[0]):
    #        for j in range(i+1,df_veza.shape[0]):
    #            aa=(df_veza.values[i]==df_veza.values[j])[0]
    #            print(aa)
    #            if aa:
    #                Se[4,df_veza.index[i],df_veza.index[j]] = 1
    #            Se[4,df_veza.index[i],df_veza.index[j]] = Se[4,df_veza.index[j],df_veza.index[i]]

    for i in range(No_class):
        for j in range(i+1,No_class):
            Mut_info = mutual_info_score(y_train_com[:45000,i].astype(int),y_train_com[:45000,j].astype(int))
            Mat = pd.crosstab(y_train_com[:,i],y_train_com[:,j])
            chi2, pvalue, dof, ex = sp.chi2_contingency(Mat)
            Se[0,i,j] = chi2
            print([chi2,pvalue])
            Se[0,j,i] = Se[0,i,j]
            Se[1,i,j] = Mut_info
            Se[1,j,i] = Se[1,i,j]  
            Se[2,i,j] = np.exp(-koef1*np.sum(np.abs(y_train_com[:,i]-y_train_com[:,j])))
            Se[2,j,i] = Se[2,i,j]
            
    R2 = np.load('Z_train_un.npy')
    scaler = StandardScaler()
    R2 = R2.reshape([R2.shape[0]*R2.shape[1],1])
    R2[R2==-np.inf] = -10
    R2[R2==np.inf] = 10
    R2[R2==-np.inf] = np.min(R2) - 10
    R2[R2==np.inf] = np.max(R2) + 10
    scaler.fit(R2)
    R2 = scaler.transform(R2)
    R2 = R2.reshape([int(R2.shape[0]/No_class),No_class])
    
    Corelation_mat = np.corrcoef(R2.T)
    Corelation_mat[Corelation_mat<0] = 0
    np.fill_diagonal(Corelation_mat,0)
    
    Se[3,:,:] = Corelation_mat
    np.save('Se',Se)
    
    Se_train = np.zeros([Noinst_train,NoGraph,No_class,No_class])
    Se_test = np.zeros([Noinst_test,NoGraph,No_class,No_class])
    
    for i in range(Noinst_train):
        Se_train[i,:,:,:] = Se
    
    for i in range(Noinst_test):
        Se_test[i,:,:,:] = Se  
    
    return Se_train, Se_test