#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Part 3 : Support Vector Machine in following the course "Machine learning A-Z" at Udemy
    The dataset can be found here https://www.superdatascience.com/machine-learning/
    
    Objective : Predict the likelihood for a person to buy an SUV in function of his age 
    and his estimated salary
    
Created on Sun Feb 25 18:49:53 2018
@author: marinechap

"""

# Import libraries 
import pandas as pd

from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm             import SVC
from sklearn.metrics         import precision_score

import matplotlib.pyplot  as plt
import Display_graph as dg

# Parameters 
name_file    = 'Social_Network_Ads.csv'
nb_indep_var =  4 

# Import dataset
dataset   = pd.read_csv(name_file)
indep_var = dataset.iloc[:,2:-1].values
dep_var   = dataset.iloc[:,nb_indep_var].values

# Split the dataset 
indep_train, indep_test, dep_train, dep_test = train_test_split(indep_var, dep_var, 
                                                                test_size = 0.25, 
                                                                random_state = 0)
# Feature scalling 
stdScal     = StandardScaler()
indep_train = stdScal.fit_transform(indep_train)
indep_test  = stdScal.transform(indep_test)

"""
 Support Vector Machine
 
 We are using different kinds of kernels to find the hyperplan (for multidimensional problem) 
 / line (in this case) which have the best maximum margin criterion.
 
 The maximum margin is the equal distance between the significant points of each category. 
 These points are support vectors. 
 
 There are different kernels which can be used but we will show 
 only the linear kernel and the polynomial kernel.
 
"""

# SVM with linear kernel
classifier_linear = SVC(kernel = 'linear')
classifier_linear.fit(indep_train, dep_train)
dep_pred_linear   = classifier_linear.predict(indep_test)

# SVM with polynomial kernel
classifier_poly = SVC(kernel = 'poly')
classifier_poly.fit(indep_train, dep_train)
dep_pred_poly   = classifier_poly.predict(indep_test)

#  Compute the precision score for each classifier

precision_poly = precision_score (dep_test, dep_pred_poly)
precision_linear = precision_score (dep_test, dep_pred_linear)

# Visualising the Training set results

plt.subplot(2,2,1)
plt = dg.display_classifier(plt, classifier_linear,indep_train, dep_train)
plt.title('Linear kernel')
plt.ylabel('Estimated Salary \n (training set)')

plt.subplot(2,2,2)
plt = dg.display_classifier(plt, classifier_poly,indep_train, dep_train)   
plt.title('Polynomial kernel')

# Visualising the Test set results

plt.subplot(2,2,3)
plt = dg.display_classifier(plt, classifier_linear,indep_test, dep_test) 
plt.xlabel('Age \n precision = %s '%(precision_linear))
plt.ylabel('Estimated Salary \n (test set)')

plt.subplot(2,2,4)
plt = dg.display_classifier(plt, classifier_poly,indep_test, dep_test)
plt.xlabel('Age \n precision = %s '%(precision_poly))

plt.suptitle('SVM algorithm \n', size = 'x-large')
plt.savefig('SVM.pdf', bbox_inches='tight')
plt.show()


