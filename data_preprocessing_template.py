# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 22:50:40 2017

@author: Ravneet
"""

#Data Preprocessing Template
# For Maths Work
import numpy as np
# For Maths Graphs and Other plots
import matplotlib.pyplot as plt
# To Import Datasets
import pandas as pd

# Importing datasets
dataset = pd.read_csv('Data.csv')
# Making Matrix of Independent variables
X = dataset.iloc[:,:-1].values
# Making Matrix of Dependent Variables
Y = dataset.iloc[:,3].values

#Spliting Datasets into Training Sets
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.3,random_state = 0)

#Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit(X_test)"""