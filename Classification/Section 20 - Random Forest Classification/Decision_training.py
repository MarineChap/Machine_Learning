#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part 3 : Decision tree classifier in following the course "Machine learning A-Z" at Udemy
    The dataset can be found here https://www.superdatascience.com/machine-learning/
    
    Objective : Predict the likelihood for a person to buy an SUV in function of his age 
    and his estimated salary
    
    This classifier are overfitting

Created on Mon Feb 26 13:27:05 2018

@author: marinechap
"""

# Import libraries 
import pandas as pd

from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier
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

# Decision tree classifier

classifier = DecisionTreeClassifier(criterion = 'entropy', random_state =0 )
classifier.fit(indep_train, dep_train)
dep_pred   = classifier.predict(indep_test)

# Decision tree classifier

classifier_forest = RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state =0 )
classifier_forest.fit(indep_train, dep_train)
dep_pred_forest   = classifier_forest.predict(indep_test)

#  Compute the precision score for each classifier

precision = precision_score (dep_test, dep_pred)
precision_forest = precision_score (dep_test, dep_pred_forest)


# Visualising the Training set results
mpl_fig = plt.figure()

plt.subplot(2,2,1)
plt = dg.display_classifier(plt, classifier,indep_train, dep_train)
plt.title('tree classifier')
plt.ylabel('Estimated Salary \n (training set)')

plt.subplot(2,2,2)
plt = dg.display_classifier(plt, classifier_forest,indep_train, dep_train)   
plt.title('Random forest classifier')

# Visualising the Test set results

plt.subplot(2,2,3)
plt = dg.display_classifier(plt, classifier,indep_test, dep_test) 
plt.xlabel('Age \n precision = %s '%(precision))
plt.ylabel('Estimated Salary \n (test set)')

plt.subplot(2,2,4)
plt = dg.display_classifier(plt, classifier_forest,indep_test, dep_test)
plt.xlabel('Age \n precision = %s '%(precision_forest))

plt.suptitle('Decision tree algorithm \n', size = 'x-large')
plt.savefig('Decision_tree_graph.png', bbox_inches='tight')

plt.show()
