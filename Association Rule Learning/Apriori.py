#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Apriori in following the course "Machine learning A-Z" at Udemy
    The dataset can be found here https://www.superdatascience.com/machine-learning/
    
    Subject : Find relation in a list of products. 

    An association rule has two parts, an antecedent (if) and a consequent (then).
    Association rules are created by analyzing data for frequent if/then patterns 
    and using the criteria support, confidence and lift to identify the most important relationships.
    
    Support(M1) = User(M1) /User(dataset)
    
    Confidence(M1, M2) = User(M1, M2) / User(M1)
    
    Lift(M1, M2) = confidence (M1, M2)/ support(M2)
    
    with M1 : antecedent 
         M2 : consequent

    
Created on Mon Mar  5 16:21:36 2018

@author: marinechap

"""

# Import libraries 
import pandas as pd
from apyori import apriori

# This function takes as argument your results list and return a tuple list with the format:
# [(rh, lh, support, confidence, lift)] 
def inspect(results):
    
    rh          = [tuple(result[2][0][0]) for result in results]
    lh          = [tuple(result[2][0][1]) for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    

    return list(zip(rh, lh, supports, confidences, lifts))



# Parameters 
name_file    = 'Market_Basket_Optimisation.csv'

# Import dataset
dataset   = pd.read_csv(name_file, header = None)


transactions =[]

for index_list in range(0, len(dataset.values)):
    transactions.append([dataset.values[index_list, index_product] for index_product in range(0, len(dataset.values[0,:])) if str(dataset.values[index_list, index_product]) != 'nan'])

"""
Apriori algorithm : 
    Step 1 : Set a minimum support and confidence 
    Step 2 : Take all the subsets in transactions having higher support than minimum support 
    Step 3: Take all the rules of theses subsets having higher confidence than minimum confidence 
    Step 4: Store the rule by decreasing lift

"""   
#  Training Apriori on the dataset 
result = list(apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2))

# this command creates a data frame to view
resultDataFrame=pd.DataFrame(inspect(result),
                columns=['rhs','lhs','support','confidence','lift'])

resultDataFrame = resultDataFrame.sort_values(by='lift', ascending = False)

"""
5 first rules : 
    
    whole wheat pasta, mineral water       -> olive oil
    milk, mineral water, frozen vegetables -> soup
    fromage blanc 				           -> honey
    light cream 					      -> chicken
    pasta 						      -> escalope


"""
    
