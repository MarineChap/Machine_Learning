#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part 2 : Polynomial regression in following the course "Machine learning A-Z" at Udemy
    
    The dataset can be found here https://www.superdatascience.com/machine-learning/
    Subject : Compute the salary of a new employee in function of his level
    
    Why choose this model ? 
    Pro : Works on any size of dataset and is better on non-linear problems
    Con : Needs to choose the good polynomial degree for a good bias/variance tradeoff
        - For this, we can use the method of backpropagation elimination. We choose a high degree
        and only interesting degrees will be kept in the model. 
        But for this, we need (with the class OLS) a dataset more consequent (< 20 data). 
    
Created on Sat Feb 24 17:30:45 2018
@author: marinechap
"""

#Import libraries 

import pandas as pd
import matplotlib.pyplot  as plt
import matplotlib.patches as mpatches
from sklearn.linear_model  import LinearRegression
from sklearn.preprocessing import PolynomialFeatures 

"""
 Preprocessing data
"""

#Parameters 
name_file    = 'Position_Salaries.csv'
nb_indep_var =  4

#Import dataset
dataset   = pd.read_csv(name_file)
indep_var = dataset.iloc[:,1:-1].values
dep_var   = dataset.iloc[:,nb_indep_var].values


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
pol = PolynomialFeatures(degree = 4)

# Transformation of the matrix of features in polynomial matrix of features
indep_poly = pol.fit_transform(indep_var)

# Linear regression with all the new independent variables
Lin_reg_poly = LinearRegression()
Lin_reg_poly.fit(indep_poly, dep_var)

# Predicting the result and visualizing the different regressions

indep_test = 6.5
dep_predict = Lin_reg_poly.predict(pol.transform(6.5))

plt.scatter(indep_var , dep_var    , color = 'red')
plt.scatter(indep_test, dep_predict, color = 'purple')

plt.plot(indep_var, Lin_reg.predict(indep_var)      , color = 'blue')
plt.plot(indep_var, Lin_reg_poly.predict(indep_poly), color = 'green')


legend1 = mpatches.Patch(color = 'red'   , label = 'Dataset')
legend2 = mpatches.Patch(color = 'blue'  , label = 'Linear regression')
legend3 = mpatches.Patch(color = 'green' , label = 'Polynomial regression (degree = 2)')
legend4 = mpatches.Patch(color = 'purple', label = 'Prediction')
plt.legend(handles = [legend1, legend2, legend3, legend4])

plt.title ('Salary vs Position')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

print('The adequat salary for the new employee (level = 6.5) is', dep_predict[0])