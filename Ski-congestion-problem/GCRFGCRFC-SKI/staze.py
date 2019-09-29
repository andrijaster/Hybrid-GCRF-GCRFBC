# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 11:40:25 2019

@author: Andrija Master
"""

import pandas as pd
import numpy as np

staze = np.load('staze.npy')
staze  = staze.astype(int)
skijasi = pd.read_csv(str(staze[0]),index_col='date1')
data = skijasi.drop(['label','vreme_pros'],axis=1)
output = skijasi.label
for i in range(1,7):
    skijasi = pd.read_csv(str(staze[i]),index_col='date1')
    out = skijasi.vreme_pros
    output = pd.concat([output,out], axis = 1)
    dl = skijasi.drop(['label','vreme_pros'],axis=1)
    data = pd.concat([data,dl], axis = 1)
    data.to_csv('atribute')
    output.to_csv('output')
    