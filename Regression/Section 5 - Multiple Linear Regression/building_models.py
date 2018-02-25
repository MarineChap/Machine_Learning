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

def multiple_regression(indep_train, dep_train):

    regressor = LinearRegression()
    regressor.fit(indep_train, dep_train)
    return regressor
    
"""
 Automatic backpropagation elimination with adjusted R_square
     Objective : to eliminate independent variables not significantly for the result. 
"""  
  
def backpropagation_elimination(indep_train, dep_train, sl)  :

    # Initialisation = all independant variable are used 

    indep_opt     = indep_train
    index_var_opt = np.arange(0,len(indep_opt[0]))
    
    # Fit the model with all possible predictors
    
    regressor         = stm.OLS(endog = dep_train, exog = indep_opt).fit()
    prev_rsquared_adj = regressor.rsquared_adj

    for i in range(len(indep_train[0])):

        # Select the independent variable with the highest p_value
        
        maxpvalue = max(regressor.pvalues).astype(float)
           
        """
         If this p_value is higher than the sl, this variable is not considered as significant
         and can be removed of the model, unless the adjusted R square isn't improve by the removal. 
         As the SL has beeen choose by humain, sometime we want remove usual variables which are above. 
         Adjusted R square is a good criterion to know if the variable is really important. 
         
         Else we have reached our threshold. The model is finished.  
        """
        if maxpvalue > sl:

            # Temporal removal of the independent variable's index which is maybe useless
            index_maxpvalue    = regressor.pvalues.argmax()
            index_var_opt_temp = np.delete(index_var_opt, index_maxpvalue, 0)
            
            # Compute the new model 
            regressor          = stm.OLS(endog = dep_train, exog = indep_opt[:,index_var_opt_temp]).fit()
            
            # Verification with the adjusted R_squared criterion of the improvement on the model
            if(regressor.rsquared_adj > prev_rsquared_adj):
                
                """
                 The improvement is shown by an increase of the criterion. 
                 We can definitely delete the variables' index from the list of useful variables.
                """
                
                index_var_opt     = index_var_opt_temp
                prev_rsquared_adj = regressor.rsquared_adj
                
            else :
                break
        else: 
            break

    regressor = stm.OLS(endog = dep_train, exog = indep_opt[:,index_var_opt]).fit()    
    return regressor, index_var_opt

"""
 Automatic forward selection with adjusted R_square
     Objective : to add only independent variables significant for the result.
"""

def forward_selection(indep_train, dep_train, sl):
     
    # initialization = all independant variable are compute to sort variables by p_value ascending. 
    regressor         = stm.OLS(endog = dep_train, exog = indep_train).fit()
    index_minpvalue   = regressor.pvalues.argsort()
    prev_rsquared_adj = regressor.rsquared_adj
    
    # Select the variable with the lowest p_value
    indep_opt     = indep_train[:,index_minpvalue[0]]
    index_var_opt = index_minpvalue[0]

    """
     We fit the model with a new predictor until the maximal pvalue of the model 
     will be higher than significant level with a decrease adjusted R_square 
    """
    
    for j in range(1,len(index_minpvalue)):
        
        index_var_opt = np.append(arr = index_var_opt, values = index_minpvalue[j])
        indep_opt     = indep_train[:,index_var_opt]
        regressor     = stm.OLS(endog = dep_train, exog = indep_opt).fit()
        maxpvalue     = max(regressor.pvalues).astype(float)

        if ((maxpvalue > sl) and(regressor.rsquared_adj < prev_rsquared_adj)) :
            break;
        prev_rsquared_adj = regressor.rsquared_adj

    # Fit the model without the last predictor
    regressor = stm.OLS(endog = dep_train, exog = indep_train[:,index_var_opt[:-1]]).fit()
    return  regressor, index_var_opt[:-1]

"""
 Quadratic normalized errors of the model
"""

def error(dep_pred, dep_test):

    error = 0
    for i in range(len(dep_test)):
        error = error + (dep_pred[i] - dep_test[i])**2 
    return np.sqrt(error/len(dep_pred))