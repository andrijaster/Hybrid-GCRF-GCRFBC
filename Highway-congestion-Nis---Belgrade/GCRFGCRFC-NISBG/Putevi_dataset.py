# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 10:42:49 2018

@author: Andrija Master
"""
import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.utils import shuffle


df = pd.read_csv("df_slobodno.csv", header = 0, index_col = 0)
df = shuffle(df)

atribute = df.iloc[:,:-36]
output = df.iloc[:,-36:]