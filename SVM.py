# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 15:16:25 2023

@author: USER
"""
# Support Vector Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
ds = pd.read_csv('thunder_data.csv')
X=ds.iloc[:, 1:-1].values
y=ds.iloc[:,-1].values


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)
print(X.shape)
print(np.reshape(X, (120, 4)))
print(y)
print(y.shape)
print(np.reshape(y, (120,)))


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training the SVR model on the whole dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

#predicting SVR Regression
y_pred=regressor.predict(y)
print(y_pred)


