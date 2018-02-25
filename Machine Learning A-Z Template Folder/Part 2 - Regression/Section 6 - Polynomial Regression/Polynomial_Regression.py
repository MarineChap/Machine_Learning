#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part 2 : Polynomial regression in following the course "Machine learning A-Z" at Udemy
    
    The dataset can be found here https://www.superdatascience.com/machine-learning/
    Subject : Compute the salary of a new employe in function of his level
    
Created on Sat Feb 24 17:30:45 2018
@author: marinechap
"""

#Import libraries 

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures 

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


# Simple Linear regression 
Lin_reg = LinearRegression()
Lin_reg.fit(indep_var, dep_var)

"""
 Polynomial regression : 
     y = b0 + b1 * x + b2 * x^2 + b3 * x^3 + ... + bn * x^n
         avec y = dependent variable 
              x = independent variable 
     
 To compute a linear regression here, we need to transform the polynomial problem in linear problem
     y = b0*x0 + b1*x1 + b2*x2 + ... + bn*xn 
         avec y = dependant variable 
              x1, x2, ..., xn = independent variables 
              
 So, we need to transform the matrix of features to have x0 = 1, x1 = x, x2 = x^2, ..., xn = x^n 
 before applied a linear regression on the independent variables. 

 The multicollinearity is still true because variables are linearly independent. 

 This working in the exact same way with multiple independent variables. 
"""

# Definition of the polynomial degree
pol = PolynomialFeatures(degree=4)

# Transformation of the matrix of features in polynomial matrix of features
indep_poly = pol.fit_transform(indep_var)

# Linear regression with all the new independent variables
Lin_reg_poly = LinearRegression()
Lin_reg_poly.fit(indep_poly, dep_var)

# Predicting the result and visualizing the different regressions

indep_test = 6.5
indep_test = pol.transform(indep_test)
dep_predict = Lin_reg_poly.predict(indep_test)

plt.scatter(indep_var, dep_var, color = 'red')
legend1 = mpatches.Patch(color='red', label='Dataset')

plt.plot(indep_var, Lin_reg.predict(indep_var), color = 'blue')
legend2 = mpatches.Patch(color='blue', label='Linear regression')

plt.plot(indep_var, Lin_reg_poly.predict(indep_poly), color = 'green')
legend3 = mpatches.Patch(color='green', label='Polynomial regression')

plt.scatter(indep_test[:,1], dep_predict, color = 'purple')
legend4 = mpatches.Patch(color='purple', label='Prediction')

plt.title('Salary vs Position')
plt.legend(handles=[legend1, legend2, legend3, legend4])
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

print('The adequat salary for the new employee (level = 6.5) is', dep_predict[0])