# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 11:50:02 2018

@author: Andrija Master
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from scipy.optimize import minimize
import scipy as sp
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import math

""" CLASS GCRFCNB """

class GCRFCNB:
    
    def __init__(self):
        pass
    
    def muKov(alfa, R, Precison, Noinst, NodeNo):
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
    
    def sigmaCal(ceta): # Provereno
        Sigma=1/(1 + np.exp(-ceta))
        Sigma[Sigma>0.99999999] = 0.99999999
        Sigma[Sigma<1e-10] = 1e-10
        return Sigma
    
    
    """ PREDICT """
    
    def predict(self,R,Se):
    
        NodeNo = Se.shape[3]
        Noinst = Se.shape[0]
        Precison = GCRFCNB.Prec(self.alfa, self.beta, NodeNo, Se, Noinst)
        mu, Kovmat = GCRFCNB.muKov(self.alfa, R, Precison, Noinst, NodeNo)
        Prob = GCRFCNB.sigmaCal(mu)
        Class = np.round(Prob,0)
        self.Prob = Prob
        self.Class = Class
        return self.Prob, self.Class

    """ FIT """  
    
    def fit(self,R,Se,Y,x0 = None, learn = 'SLSQP', maxiter = 1000, learnrate = 0.1):
    
        def L(x,Y,ModelUNNo,ModelSTNo,NodeNo,Noinst,R):
            
            alfa=x[:ModelUNNo]
            beta=x[-ModelSTNo:]
            print(alfa)
            Precison = GCRFCNB.Prec(alfa, beta, NodeNo, Se, Noinst)
            mu,kovMat = GCRFCNB.muKov(alfa,R,Precison,Noinst,NodeNo)
            sigma = GCRFCNB.sigmaCal(mu)
            L = np.sum(Y*np.log(sigma)+(1-Y)*np.log(1-sigma))
            print('skor je {}'.format(L))
            return -1*L
        
        def DLdx(x,Y,ModelUNNo,ModelSTNo,NodeNo,Noinst,R):
            
            def sigmaFUN(Y,mu):    
                sigma = GCRFCNB.sigmaCal(mu)
                sigmafun=Y-sigma
                return sigmafun
            
            def dPrecdbeta(Noinst,ModelSTNo,NodeNo,Se): # PROVERENO
                dPrecdbeta = np.zeros([Noinst,ModelSTNo,NodeNo,NodeNo])
                dPrecdbeta = -Se
                for m in range(Noinst):
                    for L in range(ModelSTNo):
                        dPrecdbeta[m,L,:,:]=2*(dPrecdbeta[m,L,:,:] + np.diag(-dPrecdbeta[m,L,:,:].sum(axis=1))) 
                return dPrecdbeta
            
            def dLdalfadbeta(sigmafun,dmudalfa,dmudbeta,ModelUNNo,ModelSTNo):
                dLdalfa = np.zeros(ModelUNNo)
                dLdbeta = np.zeros(ModelSTNo)
                for i in range(ModelUNNo):
                    dLdalfa[i] = np.sum(sigmafun*dmudalfa[:,i,:])
                for i in range(ModelSTNo):    
                    dLdbeta[i] = np.sum(sigmafun*dmudbeta[:,i,:])
                return dLdalfa,dLdbeta  
            
            def dPrecdalfa(NodeNo,ModelUNNo): # Provereno
                dPrecdalfa=np.zeros([ModelUNNo,NodeNo,NodeNo])
                dQ1dalfa=np.identity(NodeNo)
                for p in range(ModelUNNo):
                    dPrecdalfa[p,:,:]=dQ1dalfa*2
                return dPrecdalfa
            
            def dbdalfa(ModelUNNo,Noinst,R,NodeNo): # Provereno  1 
                dbdalfa = np.zeros([Noinst,ModelUNNo,NodeNo])
                for m in range(ModelUNNo):
                    dbdalfa[:,m,:] = 2*R[:,m].reshape([Noinst, NodeNo])
                return dbdalfa
            
            def dmutdalfa(dbdalfa,DPrecdalfa,Kov,ModelUNNo,Noinst,mu): # Provereno
                dmutdalfa=np.zeros([Noinst,ModelUNNo,NodeNo])
                for m in range(Noinst):
                    for p in range(ModelUNNo):
                        dmutdalfa[m,p,:]=(dbdalfa[m,p,:]-DPrecdalfa[p,:,:].dot(mu[m,:])).T.dot(Kov[m,:,:])
                return dmutdalfa
            
            def dmutdbeta(dPrecdbeta,mu,Kov,Noinst,ModelSTNo,NodeNo): # Provereno
                dmutdbeta=np.zeros([Noinst,ModelSTNo,NodeNo])
                for m in range(0,Noinst):
                    for p in range(0,ModelSTNo):
                        dmutdbeta[m,p,:]=(-dPrecdbeta[m,p,:,:].dot(mu[m,:])).T.dot(Kov[m,:,:])
                return dmutdbeta
            
            alfa=x[:ModelUNNo]
            beta=x[-ModelSTNo:]
            DPrecdalfa=dPrecdalfa(NodeNo,ModelUNNo) # Nezavisno od alfa i iteracija
            Precison = GCRFCNB.Prec(alfa, beta, NodeNo, Se, Noinst)
            DPrecdbeta = dPrecdbeta(Noinst,ModelSTNo,NodeNo,Se)
            mu,kovMat = GCRFCNB.muKov(alfa,R,Precison,Noinst,NodeNo)
            mu[np.isnan(mu)] = 0
            Dbdalfa = dbdalfa(ModelUNNo,Noinst,R,NodeNo)
#            Dbdalfa[Dbdalfa == -np.inf] = -1e12
            Dmudalfa = dmutdalfa(Dbdalfa,DPrecdalfa,kovMat,ModelUNNo,Noinst,mu)
            Dmudbeta = dmutdbeta(DPrecdbeta,mu,kovMat,Noinst,ModelSTNo,NodeNo)
            sigmafun = sigmaFUN(Y,mu)
            DLdalfa,DLdbeta = dLdalfadbeta(sigmafun,Dmudalfa,Dmudbeta,ModelUNNo,ModelSTNo)
            DLdx = -np.concatenate((DLdalfa,DLdbeta))
            print(DLdx)
            return DLdx

        ModelUNNo = R.shape[1]
        NodeNo = Se.shape[2]
        Noinst = Se.shape[0]
        ModelSTNo = Se.shape[1]
        bnd = ((1e-8,None),)*(ModelSTNo+ModelUNNo)
        if x0 is None:
            x0 = np.abs(np.random.randn(ModelUNNo + ModelSTNo))*100
        if learn == 'SLSQP':
            res = minimize(L, x0, method='SLSQP', jac=DLdx, args=(Y,ModelUNNo,ModelSTNo,NodeNo,Noinst,R),\
                       options={'disp': True,'maxiter': maxiter,'ftol': 1e-8},bounds=bnd)
            self.alfa = res.x[:ModelUNNo]
            self.beta = res.x[ModelUNNo:ModelSTNo+ModelUNNo]
        elif learn == 'TNC':
            bnd = ((1e-6,None),)*(ModelSTNo+ModelUNNo)
            res = sp.optimize.fmin_tnc(L, x0, fprime = DLdx, \
                                       args=(Y,ModelUNNo,ModelSTNo,NodeNo,Noinst,R),\
                                       bounds = bnd)
            self.alfa = res[0][:ModelUNNo]
            self.beta = res[0][ModelUNNo:ModelSTNo+ModelUNNo]   
        elif learn == 'EXP':
            x = x0
            u1 = np.log(x0)            
            for i in range(maxiter):
                dLdx = -DLdx(x,Y,ModelUNNo,ModelSTNo,NodeNo,Noinst,R)
                u1 = u1 + learnrate*x*dLdx
                x = np.exp(u1)
                L1 = -L(x,Y,ModelUNNo,ModelSTNo,NodeNo,Noinst,R)
                print('U iteciji {} DLDX je {}'.format(i,dLdx))
                print('U iteciji {} L je {}'.format(i,L1))
            self.alfa = x[:ModelUNNo]
            self.beta = x[ModelUNNo:ModelSTNo+ModelUNNo]
            self.x = x



#""" Proba na SIN podacima """
#import time
#start_time = time.time()
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
#        return Se
#
#path = 'D:\Dokumenti\Programi Python\Proba.xlsx'
#df = pd.read_excel(path)
##R = df.iloc[:,:2].values
##R=np.random.rand(5200,2)*2-1
#R = np.load('R_sinteticki.npy')
#NodeNo = 4
#Nopoint = R.shape[0]
#Noinst = np.round(Nopoint/NodeNo).astype(int)
#i1 = np.arange(NodeNo)
#i2 = np.arange(Noinst)
#Xst = np.load('Xst.npy')
#Xst =pd.DataFrame(data=Xst)
#Xst['Node'] = np.tile(i1, Noinst)
#Xst['Inst'] = np.repeat(i2,NodeNo)
#Xst = Xst.set_index(['Inst','Node'])
#connect1=np.array([[0,1],[1,2]])
#connect2=np.array([[0,1],[2,3]])
#connect=[connect1,connect2]
#NoGraph = len(connect)
##Se = np.zeros([Noinst,NoGraph,NodeNo,NodeNo])
##Se = S(connect,Se,Xst)
#Se = np.load('Se.npy')
#
#Notrain = (Noinst*0.8).astype(int)
#Notest = (Noinst*0.2).astype(int)
#
#
#mod1 = GCRFCNB()
#mod1.alfa = np.array([1,18])
#mod1.beta = np.array([0.2,0.2])
#prob, Y = mod1.predict(R,Se)  
#Se_train = Se[:Notrain,:,:,:]
#R_train = R[:Notrain*NodeNo,:]
#Y_test = Y[Notrain:Noinst,:]
#Y_train = Y[:Notrain,:]
#
#mod1.fit(R_train, Se_train, Y_train, learn = 'TNC')  
#
#R_test = R[Notrain*NodeNo:Noinst*NodeNo,:]
#Se_test = Se[Notrain:Noinst,:,:,:]
#prob2, Y2, Var = mod1.predict(R_test,Se_test)
#Prob1 = prob2.copy()
#Prob1[Y2==0] = 1 - Prob1[Y2==0]  
#Y21 =  Y2.reshape([Y2.shape[0]*Y2.shape[1]])
#Y_test1 = Y_test.reshape([Y_test.shape[0]*Y_test.shape[1]])
#probr = prob2.reshape([prob2.shape[0]*prob2.shape[1]])
#probr1 = Prob1.reshape([Prob1.shape[0]*Prob1.shape[1]])
#print('AUC je {}'.format(roc_auc_score(Y_test1,probr)))
##print('Skor je {}'.format(accuracy_score(Y21,Y_test1)))
#print('LogPRob je {}'.format(np.sum(np.log(probr1))))
#print("--- %s seconds ---" % (time.time() - start_time))

#""" Stvarni podaci Skijasi """ 

#Spom = np.load('Se.npy')
#R_train = np.load('Z_train_com.npy')
#R_test = np.load('Z_test_com.npy')
#Y_train = np.load('Y_train.npy')
#Y_test = np.load('Y_test.npy')
#Se_train_inst = np.load('Se_train.npy')
#Se_test_inst = np.load('Se_test.npy')
#
#NodeNo = 7
#Noinst_train = np.round(R_train.shape[0]/NodeNo).astype(int)
#Noinst_test = np.round(R_test.shape[0]/NodeNo).astype(int)
#
#ModelSTNo = 6
#Se_train = np.zeros([Noinst_train,ModelSTNo,NodeNo,NodeNo])
#Se_test = np.zeros([Noinst_test,ModelSTNo,NodeNo,NodeNo])
#
#for i in range(Noinst_train):
#    Se_train[i,:5,:,:] = Spom
#    
#for i in range(Noinst_test):
#    Se_test[i,:5,:,:] = Spom    
#
#Se_train[:,5,:,:] = np.squeeze(Se_train_inst)
#Se_test[:,5,:,:] = np.squeeze(Se_test_inst)
# 
#
#mod1 = GCRFCNB()
#
#
#mod1.fit(R_train, Se_train, Y_train, learn = 'SLSQP', learnrate = 6e-4, maxiter = 300)  
#
##mod1.alfa = np.array([0.1043126 , 0.06905401, 0.08689079])
##mod1.beta = np.array([1.00008728e-08, 2.88191498e+02, 1.00000563e-08, 1.00000000e-08,
##       8.74943190e+01, 3.48984028e-03])
#    
#prob2, Y2 = mod1.predict(R_test,Se_test)  
#Y2 =  Y2.reshape([Y2.shape[0]*Y2.shape[1]])
#Y_test = Y_test.reshape([Y_test.shape[0]*Y_test.shape[1]])
#prob2 = prob2.reshape([prob2.shape[0]*prob2.shape[1]])
#
#print('AUC GCRFCNB prediktora je {}'.format(roc_auc_score(Y_test,prob2)))
#print('Skor GCRFCNB prediktora je {}'.format(accuracy_score(Y2,Y_test)))
##Skor_com = np.load('Skor_com.npy')
#Skor_com_AUC = np.load('Skor_com_AUC.npy')
#print('AUC nestruktuiranih prediktora je {}'.format(Skor_com_AUC))
##print('Skor nestruktuiranih prediktora je {}'.format(Skor_com))
#print('Logprob je {}'.format(np.sum(np.log(prob2))))

#""" Stvarni podaci Debeli """ 
#
#import time
#Spom = np.load('Se.npy')
#R_train = np.load('Z_train_com.npy')
#R_train[R_train == -np.inf] = -10
#R_train[R_train == -np.inf] = np.min(R_train)-100
#R_test = np.load('Z_test_com.npy')
#R_test[R_test == -np.inf] = -10
#R_test[R_test == -np.inf] = np.min(R_test)-100
#Y_train = np.load('Y_train.npy')
#Y_test = np.load('Y_test.npy')
#for i in range(R_train.shape[1]):
#    Range = np.abs(np.max(R_train[:,i]) + np.min(R_train[:,i]))
#    faktor = int(math.log10(Range))
#    R_train[:,i] = R_train[:,i]*10**(-faktor)
#    R_test[:,i] = R_test[:,i]*10**(-faktor)
#
#NodeNo = 10
#Noinst_train = np.round(R_train.shape[0]/NodeNo).astype(int)
#Noinst_test = np.round(R_test.shape[0]/NodeNo).astype(int)
#
#ModelSTNo = 4
#Se_train = np.zeros([Noinst_train,ModelSTNo,NodeNo,NodeNo])
#Se_test = np.zeros([Noinst_test,ModelSTNo,NodeNo,NodeNo])
#
#for i in range(Noinst_train):
#    Se_train[i,:,:,:] = Spom
#    
#for i in range(Noinst_test):
#    Se_test[i,:,:,:] = Spom    
# 
#mod1 = GCRFCNB()
#
#start_time = time.time()
#mod1.fit(R_train, Se_train, Y_train, learn = 'SLSQP', learnrate = 6e-4, maxiter = 5000)  
#
#
##mod1.alfa = np.array([1-10, 1e-10, 1e-10, 3000])
##mod1.beta = np.array([1.0000000e-10, 1.0000000e-10, 1e-10, 1e-10])
#    
#prob2, Y2 = mod1.predict(R_test,Se_test)  
#Y2 =  Y2.reshape([Y2.shape[0]*Y2.shape[1]])
#Y_test = Y_test.reshape([Y_test.shape[0]*Y_test.shape[1]])
#prob2 = prob2.reshape([prob2.shape[0]*prob2.shape[1]])
#
##Y_train = Y_train.reshape([Y_train.shape[0]*Y_train.shape[1]])
#print('AUC GCRFCNB prediktora je {}'.format(roc_auc_score(Y_test,prob2)))
##print('Skor GCRFCNB prediktora je {}'.format(accuracy_score(Y2,Y_test)))
##Skor_com = np.load('Skor_com.npy')
#Skor_com_AUC = np.load('Skor_com_AUC.npy')
#print('AUC nestruktuiranih prediktora je {}'.format(Skor_com_AUC))
##print('Skor nestruktuiranih prediktora je {}'.format(Skor_com))
#print('Logprob je {}'.format(np.sum(np.log(prob2))))
#print("--- %s seconds ---" % (time.time() - start_time))