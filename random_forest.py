# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 13:13:54 2023

@author: USER
"""
# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
ds = pd.read_csv('thunder_data.csv')
print(ds.head())
print(ds.shape)
X=ds.iloc[:, 1:-1].values
y=ds.iloc[:,-1].values
print(X)
print(ds.info())

print(X.shape)
print(np.reshape(X, (120, 4)))
print(y)
print(y.shape)
print(np.reshape(y, (120,)))
print(X)




# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Training the Random Forest Regression model on the whole dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)

#print(X_train)
print(regressor.predict(np.array([[69,27.8,22.7,190.7]])))

#predicting RandomForestRegression
y_pred=regressor.predict(X)
print(y_pred)


from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(y, y_pred)
print('rmse: %.3f' %rmse)


# Visualising the Random Forest Regression results (higher resolution)
from pandas.plotting import parallel_coordinates
plt.rcParams['legend.fontsize'] = '16'
ds.plot(figsize=(10,10), fontsize=24)
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111)
ax.set_title("Thunder Prediction", fontsize=20)
ax.tick_params(axis='x', rotation=30)
ax.tick_params(axis='both', labelsize=20)
#parallel_coordinates(ds, class_column='Thunderstorm ', ax=ax)

import numpy as np
#from flask import Flask, request, render_template
import pickle
#pickle.dump(regressor,open('thunder_data.pkl','wb'))
