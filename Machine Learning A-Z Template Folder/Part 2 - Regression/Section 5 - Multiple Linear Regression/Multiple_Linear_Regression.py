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
    
    Created on Sat Feb 24 12:14:05 2018
    @author: marinechap  
"""

#Import libraries 

import pandas as pd
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
dep_pred  = bm.multiple_regression(indep_train, indep_test, dep_train)
reg_err   = bm.error(dep_pred, dep_test)

# Parameters
sl        = 0.05

# With backpropagation elimination
dep_pred         = bm.backpropagation_elimination(indep_train, indep_test, dep_train, sl)
reg_opt_back_err = bm.error(dep_pred, dep_test)

# With forward selection
dep_pred         = bm.forward_selection(indep_train, indep_test, dep_train, sl)
reg_opt_for_err  = bm.error(dep_pred, dep_test)

print('Error with all independant variables take in consideration:', reg_err)       
print('Error after optimization by backpropagation elimination:',    reg_opt_back_err)
print('Error after optimization by forward selection:',              reg_opt_for_err )
print('We can see an improvement after optimization.')


        
