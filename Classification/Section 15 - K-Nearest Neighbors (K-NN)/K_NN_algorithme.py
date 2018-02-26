#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Part 3 : K-Nearest Neighbors in following the course "Machine learning A-Z" at Udemy
    The dataset can be found here https://www.superdatascience.com/machine-learning/
    
    Objective : Predict the likelihood for a person to buy an SUV in function of his age 
    and his estimated salary
    
    Step 1 : Choose K nearest neighbors (usually 5)
    Step 2 : Take the k-NN of the new data point, according to the Euclidian distance 
    Step 3 : Count the number of NN in each category
    Conclusion : The new data point is in the category the most represented in his neighborhood.
    
Created on Sun Feb 25 18:05:12 2018
@author: marinechap

"""

# Import libraries 
import pandas as pd

from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.metrics         import precision_score, confusion_matrix

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
 K-NN Algorithm
 n_neighbors = 5 means it is the number 5 of neighbors which is chosen 
 p = 2, metric = 'minkowski' means we are using the Euclidian distance. 
                          if p =1, we are using the Manhattan distance 
"""

classifier = KNeighborsClassifier(n_neighbors = 5, p = 2, metric = 'minkowski')
classifier.fit(indep_train, dep_train)

dep_pred   = classifier.predict(indep_test)

# Print the confusion matrix and the precision score 

print(confusion_matrix(dep_test, dep_pred))
precision = precision_score (dep_test, dep_pred)

# Visualising the Training set results

plt.subplot(1,2,1)

plt = dg.display_classifier(plt, classifier, indep_train, dep_train)
    
plt.title('Training set')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')


# Visualising the Test set results
plt.subplot(1,2,2)
plt = dg.display_classifier(plt, classifier, indep_test, dep_test)
    
plt.title('Test set')
plt.xlabel('Age \n precision = %s'%(precision))
plt.suptitle('K_NN algorithm', size = 'x-large')
plt.savefig('K_NN_algo.pdf', bbox_inches='tight')
plt.show()
