# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
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

#Taking care Of Missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN",strategy="mean",axis = 0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

#Encoding categorial data
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_feature =[0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

#Spliting Datasets into Training Sets
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test 