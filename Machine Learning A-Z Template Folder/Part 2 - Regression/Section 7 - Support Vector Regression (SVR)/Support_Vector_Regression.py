#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part 2 : Support vector regression in following the course "Machine learning A-Z" at Udemy
    
    The dataset can be found here https://www.superdatascience.com/machine-learning/
    Subject : Compute the salary of a new employee in function of his level
      
Created on Sat Feb 24 19:24:24 2018
@author: marinechap
"""

#Import libraries 

import pandas as pd

import matplotlib.pyplot  as plt
import matplotlib.patches as mpatches

from sklearn.svm           import SVR
from sklearn.preprocessing import StandardScaler


"""
 Preprocessing data
"""

#Parameters 
name_file    = 'Position_Salaries.csv'
nb_indep_var =  2

#Import dataset
dataset    = pd.read_csv(name_file)
indep_var  = dataset.iloc[:,1:-1].values
dep_var    = dataset.iloc[:,nb_indep_var].values


"""
 SVR regression with 3 sort of kernels
     - Radial basis function
     - Polynomial kernel
     - Linear kernel

 /!\ The library sklearn.svm don't handle the feature scalling. 
     We need to do that before use the class SVR.
"""

# Feature scalling 
stdScal_x = StandardScaler()
stdScal_y = StandardScaler()
indep_var = stdScal_x.fit_transform(indep_var)
dep_var   = stdScal_y.fit_transform(dep_var.reshape(len(dep_var),1))

# SVR Regression with Radial basis function
regressor_rbf = SVR(kernel = 'rbf')
regressor_rbf.fit(indep_var, dep_var)

# SVR Regression with Polynomial kernel
regressor_poly = SVR(kernel = 'poly')
regressor_poly.fit(indep_var, dep_var)

# SVR Regression with linear kernel
regressor_lin = SVR(kernel = 'linear')
regressor_lin.fit(indep_var, dep_var)

# Prediction and Visualizing the result
plt.scatter(stdScal_x.inverse_transform(indep_var), stdScal_y.inverse_transform(dep_var), color = 'red')

plt.plot(stdScal_x.inverse_transform(indep_var), stdScal_y.inverse_transform(regressor_rbf.predict(indep_var)) , color = 'green')
plt.plot(stdScal_x.inverse_transform(indep_var), stdScal_y.inverse_transform(regressor_poly.predict(indep_var)), color = 'blue')
plt.plot(stdScal_x.inverse_transform(indep_var), stdScal_y.inverse_transform(regressor_lin.predict(indep_var)) , color = 'orange')

legend1 = mpatches.Patch(color = 'red'   , label = 'Dataset')
legend2 = mpatches.Patch(color = 'green' , label = 'SVR regression with Radial basis function')
legend3 = mpatches.Patch(color = 'blue'  , label = 'SVR regression with polynomial kernel')
legend4 = mpatches.Patch(color = 'orange', label = 'SVR regression with linear kernel')
plt.legend(handles=[legend1, legend2, legend3, legend4])

plt.title('Salary vs Position')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()
