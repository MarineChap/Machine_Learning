#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part 2 : Multiple linear regression in following the course "Machine learning A-Z" at Udemy
    
    The dataset can be found here https://www.superdatascience.com/machine-learning/
    Subject : Profit in function of different spend and the location for 50 companies
    
    4 assumptions which need to be verified before build a linear regression model : 
        - Linearity
        - Homoscedasticity
        - Multivariate normality
        - Lack of multicollinearity
        
    Why choose this model ? 
    Pro : Works on any size of dataset and gives informations about the relevance of features
    Con : Needs respect the linear regression assumptions 
    
    Created on Sat Feb 24 12:14:05 2018
    @author: marinechap  
"""

#Import libraries 

import pandas as pd
import numpy  as np
import building_models as bm
from sklearn.preprocessing   import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

"""
 Preprocessing data
"""

#Parameters 
name_file    = '50_Startups.csv'
nb_indep_var =  4

#Import dataset
dataset    = pd.read_csv(name_file)
indep_var  = dataset.iloc[:,:-1].values
dep_var    = dataset.iloc[:,nb_indep_var].values

# Encoding category variable
labelEncoder   = LabelEncoder()
indep_var[:,3] = labelEncoder.fit_transform(indep_var[:,3])
hotEncoder     = OneHotEncoder(categorical_features = [3])
indep_var      = hotEncoder.fit_transform(indep_var).toarray()

"""
Avoid dummy variables trap      
    when we create dummy variable, we obtain a variable dependent of others. 
    We fail the last assumption "lack of multicollinearity." 
    To avoid this trap, we need always need to remove one of the dummy variables. 
"""
indep_var = indep_var[:,1:]

# Split dataset
indep_train, indep_test, dep_train, dep_test = train_test_split(indep_var, dep_var, test_size = 0.2, random_state = 0)

"""
 Multiple linear regression
"""
  
# With all independant variables
regressor = bm.multiple_regression(indep_train, dep_train)
dep_pred  = regressor.predict(indep_test)
reg_err   = bm.error(dep_pred, dep_test)

# Parameters
sl        = 0.05

# Add the constant independant variable (=1) in the dataset
indep_train = np.append(arr = np.ones((len(indep_train),1)).astype(int), values = indep_train,axis = 1)
indep_test  = np.append(arr = np.ones((len(indep_test ),1)).astype(int), values = indep_test, axis = 1)
  
# With backpropagation elimination
regressor, index_var_opt = bm.backpropagation_elimination(indep_train, dep_train, sl)
dep_pred = regressor.predict(indep_test[:,index_var_opt]) 
reg_opt_back_err = bm.error(dep_pred, dep_test)


# With forward selection
regressor, index_var_opt = bm.forward_selection(indep_train, dep_train, sl)
dep_pred = regressor.predict(indep_test[:,index_var_opt]) 
reg_opt_for_err  = bm.error(dep_pred, dep_test)


print('Error with all independant variables take in consideration:', reg_err)       
print('Error after optimization by backpropagation elimination:',    reg_opt_back_err)
print('Error after optimization by forward selection:',              reg_opt_for_err )
print('We can see an improvement after optimization.')