# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp

plt.close('all')

def statistiku2(pomocna, stara, brojminuta):
    for j in range(pomocna.shape[0]):
        pomocna1 = stara.loc[pomocna.index[j] - pd.Timedelta(minutes = brojminuta):pomocna.index[j]] 
        pomocna2 = pomocna1[pomocna1['location1']==pomocna1['location2']]
        grupisana1 = pomocna1.groupby('location1')['skier']
        grupisana2 = pomocna2.groupby('location1')['deltaTimeSecs']
        stat1 = grupisana1.count()
        stat2 = grupisana1.nunique()
        stat3 = grupisana2.mean()
        a = skijasi_novo.iloc[j]
        a.loc['broj'].loc[skijasi_novo.iloc[j]['broj'].index.isin(stat1.index)]=stat1[:]
        skijasi_novo.iloc[j] = a[:]
        a = skijasi_novo.iloc[j]
        a.loc['broj_unique'].loc[skijasi_novo.iloc[j]['broj_unique'].index.isin(stat2.index)]=stat2[:]
        skijasi_novo.iloc[j] = a[:]
        a = skijasi_novo.iloc[j]
        a.loc['vreme_pros'].loc[skijasi_novo.iloc[j]['vreme_pros'].index.isin(stat3.index)]=stat3[:]
        skijasi_novo.iloc[j] = a[:]
    return skijasi_novo

def statistiku1(pomocna,stara,brojminuta):
    for j in range(pomocna.shape[0]):
        pomocna1 = stara.loc[pomocna.index[j] - pd.Timedelta(minutes = brojminuta):pomocna.index[j]]
        grupisana = pomocna1.groupby('location2')['brzina']
        stat3 = grupisana.mean()
        stat4 = grupisana.median()
        stat5 = grupisana.quantile(0.25)
        stat6 = grupisana.quantile(0.75)
        stat7 = grupisana.quantile(1)
        stat8 = grupisana.quantile(0.1)        
        stat9 = grupisana.quantile(0.9) 
        stat10 =  grupisana.aggregate(sp.stats.kurtosis)
        stat11 =  grupisana.aggregate(sp.stats.skew)
        a = skijasi_novo.iloc[j]
        a.loc['brzinaMEAN'].loc[skijasi_novo.iloc[j]['brzinaMEAN'].index.isin(stat3.index)]=stat3[:]
        skijasi_novo.iloc[j] = a[:]
        a = skijasi_novo.iloc[j]
        a.loc['brzinaMEDIAN'].loc[skijasi_novo.iloc[j]['brzinaMEDIAN'].index.isin(stat4.index)]=stat4[:]
        skijasi_novo.iloc[j] = a[:]
        a = skijasi_novo.iloc[j]
        a.loc['brzinaPERCENTILE25'].loc[skijasi_novo.iloc[j]['brzinaPERCENTILE25'].index.isin(stat5.index)]=stat5[:]
        skijasi_novo.iloc[j] = a[:]
        a = skijasi_novo.iloc[j]
        a.loc['brzinaPERCENTILE75'].loc[skijasi_novo.iloc[j]['brzinaPERCENTILE75'].index.isin(stat6.index)]=stat6[:]
        skijasi_novo.iloc[j] = a[:]
        a = skijasi_novo.iloc[j]
        a.loc['brzinaPERCENTILE10'].loc[skijasi_novo.iloc[j]['brzinaPERCENTILE10'].index.isin(stat7.index)]=stat7[:]
        skijasi_novo.iloc[j] = a[:]
        a = skijasi_novo.iloc[j]
        a.loc['brzinaPERCENTILE100'].loc[skijasi_novo.iloc[j]['brzinaPERCENTILE100'].index.isin(stat8.index)]=stat8[:]
        skijasi_novo.iloc[j] = a[:]
        a = skijasi_novo.iloc[j]
        a.loc['brzinaPERCENTILE90'].loc[skijasi_novo.iloc[j]['brzinaPERCENTILE90'].index.isin(stat9.index)]=stat9[:]
        skijasi_novo.iloc[j] = a[:]
        a = skijasi_novo.iloc[j]
        a.loc['brzinaKURT'].loc[skijasi_novo.iloc[j]['brzinaKURT'].index.isin(stat10.index)]=stat10[:]
        skijasi_novo.iloc[j] = a[:]
        a = skijasi_novo.iloc[j]
        a.loc['brzinaSKEW'].loc[skijasi_novo.iloc[j]['brzinaSKEW'].index.isin(stat11.index)]=stat11[:]
        skijasi_novo.iloc[j] = a[:]
    return skijasi_novo
        
        
vreme = pd.read_csv('Vreme.csv')
vreme['date1']=pd.to_datetime(vreme['date1'])
vreme = vreme.set_index(['date1'])


granice = np.load('tmax.npy')
staze = np.load('staze.npy')

brojminuta = 5
no_steps = 9

skijasi = pd.read_csv('Skijasi_novo')
skijasi = skijasi[skijasi['deltaTimeSecs']!=0]
skijasi['date1'] = pd.to_datetime(skijasi['date1'])
unique = set(skijasi['location1'])
skijasi = skijasi.set_index(['date1'])
skijasi['broj'] = np.zeros(skijasi.shape[0])
df_list = []
skijasi['date1pom'] = pd.to_datetime(skijasi.index)
skijasi['date1pom'] = skijasi['date1pom'].apply(lambda x: x.strftime('%Y-%m-%d'))
skijasi.sort_index(axis=0,inplace=True)
uniquedate = set(skijasi['date1pom'])
start_date = ' 09:00:00'
end_date = ' 17:00:00'
tabela = []
for i in uniquedate:
    start= i + start_date
    end = i + end_date
    start1 = pd.to_datetime(start)
    end1 = pd.to_datetime(end)
    indeksi = pd.date_range(start = start1, end = end1, freq = '5min')
    pomocna = pd.DataFrame(np.zeros([len(indeksi),1]),columns = ['A'],index=indeksi)
    tabela.append(pomocna)
pomocna = pd.concat(tabela,axis=0)
pomocna.index.name = 'date'
pomocna['date1pom'] = pomocna.index
pomocna['date1pom'] = pd.to_datetime(pomocna['date1pom']).apply(lambda x: x.strftime('%Y-%m-%d %H'))
pomocna['date1pom'] = pd.to_datetime(pomocna['date1pom'])
pomocna = pd.merge(pomocna,vreme,how='inner',left_on = 'date1pom', right_index  = True)
pomocna.drop(['A','date1pom'],axis=1,inplace=True)
pomocna.sort_index(axis=0,inplace=True)
pomocna['vreme_pros'] = np.zeros([len(pomocna),1])
pomocna['broj'] = np.zeros([len(pomocna),1])
pomocna['broj_unique'] = np.zeros([len(pomocna),1])
pomocna['broj_unique'] = np.zeros([len(pomocna),1])
pomocna['brzinaMEAN'] = np.zeros([len(pomocna),1])
pomocna['brzinaMEDIAN'] = np.zeros([len(pomocna),1])
pomocna['brzinaPERCENTILE25'] = np.zeros([len(pomocna),1])
pomocna['brzinaPERCENTILE75'] = np.zeros([len(pomocna),1])
pomocna['brzinaPERCENTILE10'] = np.zeros([len(pomocna),1])
pomocna['brzinaPERCENTILE100'] = np.zeros([len(pomocna),1])
pomocna['brzinaPERCENTILE90'] = np.zeros([len(pomocna),1])
pomocna['brzinaKURT'] = np.zeros([len(pomocna),1])
pomocna['brzinaSKEW'] = np.zeros([len(pomocna),1])
pomocna['hour-minute']  = pomocna.index.hour*60 + pomocna.index.minute 
pomocna['label'] = np.zeros([len(pomocna),1])



i=0
skijasi_novo = []
for k in unique:
    pomocna['location'] = k*np.ones([len(pomocna),1])
    pomocna['location'] = pomocna['location'].astype(int) 
    skijasi_novo.append(pomocna[:])
    i += 1

skijasi_novo = pd.concat(skijasi_novo)
skijasi_novo['date1']=skijasi_novo.index
skijasi_novo.set_index(['date1','location'],inplace=True)
skijasi_novo.sort_index(inplace=True)
skijasi_novo = skijasi_novo.unstack()
i=0


    
skijasi_novo = statistiku2(skijasi_novo,skijasi,brojminuta)

skijasi2 = skijasi
skijasi2['date2'] = pd.to_datetime(skijasi2['date2'])
skijasi2['brzina']= np.abs(skijasi['deltaVerticalMeters']/skijasi['deltaTimeSecs'])
skijasi2.set_index('date2',inplace=True)
skijasi2.sort_index(axis=0,inplace=True)
skijasi_novo = statistiku1(skijasi_novo,skijasi2,brojminuta)

skijasi_novo.columns = skijasi_novo.columns.swaplevel(0,1)
skijasi_novo.sortlevel(0, axis=1, inplace=True)

df_list = []

i=0
for k in staze:
    wherelab = skijasi_novo[k].loc[:,'vreme_pros']>0
    skijasi_novo[k].loc[:,'label'][wherelab] = 1
    skijasi_novo[k].loc[:,'label'] = skijasi_novo[k].loc[:,'label'].shift(-no_steps).fillna(0)
    skijasi_novo[k].loc[:,'vreme_pros'] = skijasi_novo[k].loc[:,'vreme_pros'].shift(-no_steps).fillna(0)
    print(skijasi_novo[k][skijasi_novo[k]['label']==0].shape,granice[i])
    df_list.append(skijasi_novo[k])
    df_list[i].to_csv(str(k))
    plt.figure(i)
    skijasi_novo[k].loc[:,'broj'].plot(kind='hist')
    i+=1

