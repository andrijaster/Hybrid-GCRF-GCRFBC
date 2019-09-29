# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 13:04:06 2018

@author: Andrija Master
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from scipy.optimize import minimize
import scipy as sp
from sklearn.metrics import mean_squared_error

""" GCRF CLASS """
class GCRF:
    
    def __init__(self):
        pass
    
    def muKov(alfa,R,Precison,Noinst,NodeNo):
        mu = np.zeros([Noinst,NodeNo])
        bv = 2*np.matmul(R,alfa)
        bv = bv.reshape([Noinst,NodeNo])
        Kov = np.linalg.inv(Precison)
        for m in range(Noinst):
            mu[m,:] = Kov[m,:,:].dot(bv[m,:])
        return mu,Kov
        
    def Prec(alfa,beta,NodeNo,Se,Noinst):
        alfasum = np.sum(alfa)
        Q1 = np.identity(NodeNo)*alfasum
        Q2 = np.zeros([Noinst,NodeNo,NodeNo])
        Prec = np.zeros([Noinst,NodeNo,NodeNo])
        pomocna = np.zeros(Se.shape)
        for j in range(Se.shape[1]):
            pomocna[:,j,:,:] = Se[:,j,:,:] * beta[j]
        Q2 = -np.sum(pomocna,axis = 1)
        for m in range(Noinst):
            Prec[m,:,:] = 2*(Q2[m,:,:]+np.diag(-Q2[m,:,:].sum(axis=0))+Q1)
        return Prec
    
# PREDICT - INFERENCE   
    def predict(self,R,Se):
        
        NodeNo = Se.shape[3]
        Noinst = np.round(R.shape[0]/NodeNo).astype(int)
        Precison = GCRF.Prec(self.alfa, self.beta, NodeNo, Se, Noinst)
        mu, kov = GCRF.muKov(self.alfa,  R, Precison, Noinst, NodeNo)
        self.prediction = mu
        self.kovarijaciona = kov
        return self.prediction

# FIT - LEARN    
    def fit(self,R,Se,y,x0 = None,learn = 'TNC', maxiter = 1000, learnrate = 0.01):
        
        def dLdX(x,ModelUNNo,NoGraph,NodeNo,Noinst,R,Se,y):
            
            def Trace(x,y): # Provereno 2
                i1,j1 = x.shape
                trMat = 0
                for k in range(i1):
                    trMat = trMat+x[k,:].dot(y[:,k])
                return trMat
            
            def dPrecdalfa(NodeNo,ModelUNNo): # Provereno 2
                dPrecdalfa = np.zeros([ModelUNNo,NodeNo,NodeNo])
                dQ1dalfa = np.identity(NodeNo)
                for p in range(ModelUNNo):
                    dPrecdalfa[p,:,:] = dQ1dalfa*2
                return dPrecdalfa
        
            def dbdalfa(ModelUNNo,Noinst,R,NodeNo): # Provereno  1 
                dbdalfa = np.zeros([Noinst,ModelUNNo,NodeNo])
                for m in range(ModelUNNo):
                    dbdalfa[:,m,:] = 2*R[:,m].reshape([Noinst, NodeNo])
                return dbdalfa

            
            def dPrecdbeta(Noinst,NoGraph,NodeNo,Se): # Proveriti gradient chekom
                dPrecdbeta = np.zeros([Noinst,NoGraph,NodeNo,NodeNo])
                dPrecdbeta = -Se
                for m in range(Noinst):
                    for L in range(NoGraph):
                        dPrecdbeta[m,L,:,:]=2*(dPrecdbeta[m,L,:,:] + np.diag(-dPrecdbeta[m,L,:,:].sum(axis=1))) 
                return dPrecdbeta
            
            def dLdbeta(y, NoGraph, Noinst, mu,Kov, Prec, dPrecdbeta): # Provereno 
                DLdbeta=np.zeros(NoGraph)
                for k in range(NoGraph):
                    for i in range(Noinst):
                        DLdbeta[k] = -1/2*(y[i,:] + mu[i,:]).T.dot(dPrecdbeta[i,k,:,:]).dot(y[i,:] - mu[i,:]) \
                        + 1/2*Trace(Kov[i,:,:],dPrecdbeta[i,k,:,:]) + DLdbeta[k]
                return -1*DLdbeta
            
            def dLdalfa(y, ModelUNNo, Noinst, dPrecdalfa, mu, Kov, dbdalfa): # Provereno 
                DLdalfa=np.zeros(ModelUNNo)
                for k in range(ModelUNNo):
                    for i in range(Noinst):
                        DLdalfa[k] = - 1/2*(y[i,:] - mu[i,:]).T.dot(dPrecdalfa[k,:,:]).dot(y[i,:] - mu[i,:]) \
                        + (dbdalfa[i,k,:].T - mu[i,:].T.dot(dPrecdalfa[k,:,:])).dot(y[i,:] - mu[i,:]) \
                        + 1/2*Trace(Kov[i,:,:],dPrecdalfa[k,:,:]) + DLdalfa[k]
                return -1*DLdalfa
            
            alfa = x[:ModelUNNo]
            beta = x[ModelUNNo:]
            Precison = GCRF.Prec(alfa, beta, NodeNo, Se, Noinst)
            mu, Kov = GCRF.muKov(alfa,  R, Precison, Noinst, NodeNo)
            DPrecdbeta = dPrecdbeta(Noinst,NoGraph,NodeNo,Se)
            DPrecdalfa = dPrecdalfa(NodeNo,ModelUNNo)
            Dbdalfa = dbdalfa(ModelUNNo,Noinst,R,NodeNo)
            DLdbeta = dLdbeta(y, NoGraph, Noinst, mu, Kov, Precison, DPrecdbeta)
            DLdalfa = dLdalfa(y, ModelUNNo, Noinst, DPrecdalfa, mu, Kov, Dbdalfa)
            DLdx = np.concatenate((DLdalfa,DLdbeta))            
            return DLdx
        
        def L(x, ModelUNNo,NoGraph,NodeNo,Noinst,R,Se,y): 
            alfa = x[:ModelUNNo]
            beta = x[ModelUNNo:]
            Precison = GCRF.Prec(alfa,beta,NodeNo,Se,Noinst)
            mu, Kov = GCRF.muKov(alfa,R,Precison,Noinst,NodeNo)
            L=0
            for i in range(Noinst):
                    L = - 1/2*(y[i,:] - mu[i,:]).T.dot(Precison[i,:,:]).dot(y[i,:] - mu[i,:]) \
                    + 1/2*np.log(np.linalg.det(Precison[i,:,:])) + L
            return -1*L
            
        ModelUNNo = R.shape[1]
        NodeNo = Se.shape[2]
        Noinst = Se.shape[0]
        NoGraph = Se.shape[1]
        if x0 == None:
            x0 = np.abs(np.random.randn(ModelUNNo + NoGraph))*1
        
        if learn == 'TNC':
            bnd = ((1e-8,None),)*(NoGraph+ModelUNNo)
            res = minimize(L, x0, method='TNC', jac=dLdX, args=(ModelUNNo,NoGraph,NodeNo,Noinst,R,Se,y), options={'disp': True,'maxiter':300},bounds=bnd)
            self.alfa = res.x[:ModelUNNo]
            self.beta = res.x[ModelUNNo:]
        elif learn == 'EXP':
            x = x0
            u1 = np.log(x0)            
            for i in range(maxiter):
                DLdx = -dLdX(x,ModelUNNo,NoGraph,NodeNo,Noinst,R,Se,y)
                u1 = u1 + learnrate*x*DLdx
                x = np.exp(u1)
                print(x)
                
            self.alfa = x[:ModelUNNo] 
            self.beta = x[ModelUNNo:]
  
#""" PROBA NA SIN. PODACIMA """
#
#def S(connect,Se,Xst):
#        for j in range(NoGraph):
#            for k,l in connect[j]:
#                if j == 0:
#                    Se[:,j,k,l] = np.exp(np.abs(Xst.iloc[:,j].unstack().values[:,k] - 
#                      Xst.iloc[:,j].unstack().values[:,l]))*0.1 
#                    Se[:,j,l,k] = Se[:,j,k,l]
#                elif j == 1:
#                     Se[:,j,k,l] = np.exp(np.abs(Xst.iloc[:,j].unstack().values[:,k] - 
#                      Xst.iloc[:,j].unstack().values[:,l]))*0.3
#                     Se[:,j,l,k] = Se[:,j,k,l]
#                    
#        return Se
#
#path = 'Proba.xlsx'
#df = pd.read_excel(path)
#R = df.iloc[:,:2].values
#NodeNo = 4
#Nopoint = R.shape[0]
#Noinst = np.round(Nopoint/NodeNo).astype(int)
#i1 = np.arange(NodeNo)
#i2 = np.arange(Noinst)
#Xst = df.iloc[:,2:]
#Xst['Node'] = np.tile(i1, Noinst)
#Xst['Inst'] = np.repeat(i2,NodeNo)
#Xst = Xst.set_index(['Inst','Node'])
#connect1=np.array([[0,1],[1,2]])
#connect2=np.array([[0,1],[2,3]])
#connect=[connect1,connect2]
#NoGraph = len(connect)
#Se = np.zeros([Noinst,NoGraph,NodeNo,NodeNo])
#Se = S(connect,Se,Xst)
#
#
#mod1 = GCRF()
#mod1.alfa = np.array([0.8,0.5])
#mod1.beta = np.array([5,22])
#vrednosti = mod1.predict(R,Se)
#
#mod1.fit(R,Se,vrednosti, learn = 'TNC')
#vrednosti1 = mod1.predict(R,Se)
#broj = vrednosti.shape[0]*vrednosti.shape[1]
#print('MSE score je {} '.format(mean_squared_error(vrednosti.reshape(broj),vrednosti1.reshape(broj))))
#print('MSE score je {} '.format(mean_squared_error(vrednosti.reshape(broj),R[:,0])))
#print('MSE score je {} '.format(mean_squared_error(vrednosti.reshape(broj),R[:,1])))