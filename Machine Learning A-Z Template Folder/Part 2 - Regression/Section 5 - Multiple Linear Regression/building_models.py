#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 12:14:05 2018
@author: marinechap
"""

import numpy as np
import statsmodels.formula.api as stm

from sklearn.linear_model import LinearRegression


""" 
 Multiple linear regression 
 
 y = b0 + b1*x1 + b2*x2 + ... + bn*xn 
     avec y = dependant variable 
          x1, x2, ..., xn = independent variable 

 There are two libraries which can do that in different ways 
     - sklearn.linear_model with the class LinearRegression : very easy to use 
       but without the computation of pvalue and other interesting statically values
     
     - statsmodels.formula.api with the class OLS : more information is computed but we need 
       to add a constant independent variable to take in consideration the bo coefficient
"""

def multiple_regression(indep_train, indep_test, dep_train):

    regressor = LinearRegression()
    regressor.fit(indep_train, dep_train)
    dep_pred  = regressor.predict(indep_test)
    return dep_pred 

"""
 Automatic backpropagation elimination 
     Objective : to eliminate independent variables not significantly for the result. 
"""  
  
def backpropagation_elimination(indep_train, indep_test, dep_train, sl)  :

    # Add the constant independant variable (=1) in the dataset
    indep_train = np.append(arr = np.ones((len(indep_train),1)).astype(int), values = indep_train,axis = 1)
    indep_test  = np.append(arr = np.ones((len(indep_test ),1)).astype(int), values = indep_test, axis = 1)
  
    # Initialisation = all independant variable are used 

    indep_opt     = indep_train
    index_var_opt = np.arange(0,len(indep_opt[0]))
    
    for i in range(len(indep_test[0])):
        
        """
         Fit the model with all possible predictors and consider selecting the independent variable 
         with the highest p_value
        """
        regressor = stm.OLS(endog = dep_train, exog = indep_opt).fit()
        maxpvalue = max(regressor.pvalues).astype(float)
    
        """
         If this p_value is higher than the sl, this variable is not considered as significant
         and can be removed of the model. 
         Else we have reached our threshold. The model is finished.  
        """
        if maxpvalue > sl:
            index_maxpvalue = regressor.pvalues.argmax()
            indep_opt       = np.delete(indep_opt,     index_maxpvalue, 1)
            index_var_opt   = np.delete(index_var_opt, index_maxpvalue, 0)
        else: 
            break;
            
    dep_pred = regressor.predict(indep_test[:,index_var_opt])
    
    return dep_pred

"""
 Automatic forward selection 
     Objective : to add only independent variables significant for the result.
"""

def forward_selection(indep_train, indep_test, dep_train, sl):
    
    # Add the constant independant variable (=1) in the dataset
    indep_train = np.append(arr = np.ones((len(indep_train),1)).astype(int), values = indep_train,axis = 1)
    indep_test  = np.append(arr = np.ones((len(indep_test ),1)).astype(int), values = indep_test, axis = 1)
  
    
    # initialization = all independant variable are compute to sort variables by p_value ascending. 
    regressor       = stm.OLS(endog = dep_train, exog = indep_train).fit()
    index_minpvalue = regressor.pvalues.argsort()
    
    # Select the variable with the lowest p_value
    indep_opt       = indep_train[:,index_minpvalue[0]]
    index_var_opt   = index_minpvalue[0]

    """
     We fit the model with a new predictor until the maximal pvalue of the model 
     will be higher than significant level
    """
    
    for j in range(1,len(index_minpvalue)):
        
        index_var_opt = np.append(arr = index_var_opt, values = index_minpvalue[j])
        indep_opt     = indep_train[:,index_var_opt]
        regressor     = stm.OLS(endog = dep_train, exog = indep_opt).fit()
        maxpvalue     = max(regressor.pvalues).astype(float)

        if maxpvalue > sl:
            break;

    # Fit the model without the last predictor
    regressor = stm.OLS(endog = dep_train, exog = indep_train[:,index_var_opt[:-1]]).fit()
    dep_pred                  = regressor.predict(indep_test [:,index_var_opt[:-1]])
    return  dep_pred 

"""
 Quadratic normalized errors of the model
"""

def error(dep_pred, dep_test):
    
    # Compute the error 
    error = 0
    for i in range(len(dep_test)):
        error = error + (dep_pred[i] - dep_test[i])**2 
    return np.sqrt(error/len(dep_pred))