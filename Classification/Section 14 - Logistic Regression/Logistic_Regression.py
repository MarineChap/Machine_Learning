#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part 3 : Logistic regression in following the course "Machine learning A-Z" at Udemy
    The dataset can be found here https://www.superdatascience.com/machine-learning/
    
    Objective : Predict the likelihood for a person to buy a SUV in function of his age 
    and his estimated salary
    
    It is a classification problem : We don't want to predict a result 
    but we want a probability to be in a category
    
    ln(p/(1-p)) = b0 + b1 * x + b2 * x2
        with : p = the probability of the answer yes between 0 and 1
               x1, x2 = independent variable
               
Created on Sun Feb 25 15:57:14 2018
@author: marinechap
"""

# Import libraries 
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, confusion_matrix


import matplotlib.pyplot  as plt
from matplotlib.colors import ListedColormap

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
 Logistic regression 
"""

classifier = LogisticRegression()
classifier.fit(indep_train, dep_train)

dep_pred   = classifier.predict(indep_test)

"""
Print the confusion matrix 
             estimate 
             yes  no      The objective is to have the maximum in a and d 
predict yes   a    b      and the minimal in c and b which represent false results
        no    c    d
       
"""

print(confusion_matrix(dep_test, dep_pred))
print(precision_score(dep_test, dep_pred))

# Visualising the Training set results

X_set, y_set = indep_train, dep_train

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), 
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), 
                label = j)
    
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results

X_set, y_set = indep_test, dep_test

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), 
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), 
                label = j)
    
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
