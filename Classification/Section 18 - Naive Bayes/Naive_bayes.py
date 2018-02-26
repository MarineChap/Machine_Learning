#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part 3 : Naive Bayes in following the course "Machine learning A-Z" at Udemy
    The dataset can be found here https://www.superdatascience.com/machine-learning/
    
    Objective : Predict the likelihood for a person to buy an SUV in function of his age 
    and his estimated salary
    
    Why is naive ? Because of his assumption 
        - Variables (Age and Salary) are independent and equally important 
        (In reality, there is probably a correlation)
    
    This algo is based on the bayes theorem : P(C|X) = P(X|C) * P(C) / P(X)
                                                with X : representing the feature (observations)
                                                     C : representing a category 
    Here, we have in this dataset two categories : Walker's people (W) and Driver people (D) 
    
    If we want to classify a new data point : 
        
    Step 1 : We compute Bayes theorem for each category C. 
    
    Hint : As we talked about probabilities here, you need to compute N-1 categories. 
    (The last is equal to 1- sum(prob of other categories))
    
        - P(C)= nbr category / nbr observation (named prior probability)
        
	   - P(X) = nbr similar observation / nbr of observation (named marginal likelihood)
           It is the probability of an observation to be in a radius around the new point
		  => Radius need to be fixed by static calculi.
		   Finally, not very useful to compute this because is always identical for each variable.
            It doesn't affect the final comparison. 
						
	   - P(X|C) = nbr similar observation of C / nbr observation of C (named likelihood)
            It is the probability for a similar observation to be in the category
		    => Radius need to be fixed 
    
    Step 2: Comparison of probabilities
    Step 3: Choose the category with the higher probability. 
    
Created on Sun Feb 25 23:33:56 2018
@author: marinechap
"""

# Import libraries 
import pandas as pd

from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes     import GaussianNB
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


# Naive Bayes

classifier = GaussianNB()
classifier.fit(indep_train, dep_train)
dep_pred   = classifier.predict(indep_test)

#  Compute the precision score for each classifier

precision = precision_score (dep_test, dep_pred)

# Visualising the Training set results

plt.subplot(2,1,1)
plt = dg.display_classifier(plt, classifier,indep_train, dep_train)
plt.title('Linear kernel')
plt.ylabel('Estimated Salary \n (training set)')

# Visualising the Test set results

plt.subplot(2,1,2)
plt = dg.display_classifier(plt, classifier,indep_test, dep_test) 
plt.xlabel('Age \n precision = %s '%(precision))
plt.ylabel('Estimated Salary \n (test set)')


plt.suptitle('Naive bayes algorithm \n', size = 'x-large')
plt.savefig('Naive_bayes_graphe.pdf', bbox_inches='tight')
plt.show()


