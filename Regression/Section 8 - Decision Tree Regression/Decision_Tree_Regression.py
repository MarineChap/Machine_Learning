#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part 2 : Decision Tree regression in following the course "Machine learning A-Z" at Udemy
    
    The dataset can be found here https://www.superdatascience.com/machine-learning/
    Subject : Compute the salary of a new employee in function of his level
    
    Why choose this model ? 
    Pro : Interpretability 
          No need for features scaling
          work on both linear / non-linear problem.
          
    Con : Poor result on too small datasets
          Overfitting can easily occur
      
Created on Sun Feb 25 00:24:24 2018
@author: marinechap
"""

#Import libraries 

import pandas as pd
import numpy  as np
import matplotlib.pyplot  as plt
import matplotlib.patches as mpatches

from sklearn.tree import DecisionTreeRegressor

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

# Decision tree regression

regressor = DecisionTreeRegressor(criterion = 'mse', random_state= 0)
regressor.fit(indep_var, dep_var)

# Predicting the result and visualizing the different regressions

indep_test  = np.arange(min(indep_var), max(indep_var), 0.1)
indep_test  = indep_test.reshape(len(indep_test), 1)
dep_predict = regressor.predict(indep_test)

plt.plot(indep_test, dep_predict, color = 'blue')
plt.scatter(indep_var, dep_var               , color = 'red')
plt.scatter(6.5      , regressor.predict(6.5), color = 'orange')

legend1 = mpatches.Patch(color = 'red'   , label = 'Dataset')
legend2 = mpatches.Patch(color = 'blue'  , label = 'Decision tree regression')
legend3 = mpatches.Patch(color = 'orange', label = 'Prediction')

plt.legend(handles = [legend1, legend2, legend3])

plt.title ('Salary vs Position')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

print('The adequat salary for the new employee (level = 6.5) is', regressor.predict(6.5)[0])