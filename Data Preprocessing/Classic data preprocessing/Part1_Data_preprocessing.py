# -*- coding: utf-8 -*-
"""
    Part 1 : Data preprocessing in following the course "Machine learning A-Z" at Udemy
    The dataset can be found here https://www.superdatascience.com/machine-learning/
    
    Created on Sat Feb 24 12:14:05 2018
    @author: marinechap 
"""

# Import libraries 
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Parameters 
name_file    = 'Data.csv'
nb_indep_var =  3 

# Import dataset
dataset   = pd.read_csv(name_file)
indep_var = dataset.iloc[:,:-1].values
dep_var   = dataset.iloc[:,nb_indep_var].values

"""
How handle missing data in the dataset
    3 methods of replacement : 
    - Mean value of the column
    - Median value of the column
    - Most current value in the column

If there are missing data in the categorical data, it is a little more complicated. 
But we can use a classifier to replace the value or remove the line 
(this last idea is not a good idea because there is an important risk to lose important data).
"""
imputer          = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
indep_var[:,1:3] = imputer.fit_transform(indep_var[:,1:3])

"""
How encoded categorical data      
    with labelEncoder when the category can have a weighty relationship between them     
    with OneHotEncoder when we want to keep categories independent by creating dummy variables.
"""
labelEncoder   = LabelEncoder()
dep_var        = labelEncoder.fit_transform(dep_var)

indep_var[:,0] = labelEncoder.fit_transform(indep_var[:,0])
hotEncoder     = OneHotEncoder(categorical_features = [0])
indep_var      = hotEncoder.fit_transform(indep_var).toarray()

"""
How split the dataset in train set and tests set ? 
    The test size define the repartition of the data. We don't want too many datas in the train 
    because we have a risk of overfitting. Usually, test_size is fixed below 0.4
"""
indep_train, indep_test, dep_train, dep_test = train_test_split(indep_var, dep_var, test_size = 0.2, random_state = 0)

"""
Feature scalling when variables are not at the same scale

    Why ? Because the most part of the machine learning algorithms are based on the Euclidian distance. 
    If a variable has a range more important, the mathematical computation will give lots of more importance to this variable. 
    
    For this, we can use two methods : 
        - Standardization 
        - Normalization 
    
    This allows to the algorithm to converge faster. We don't always need to do this 
    because the most part of libraries in machine learning take directly care of this problem. 
"""
stdScal     = StandardScaler()
indep_train = stdScal.fit_transform(indep_train)
indep_test  = stdScal.transform(indep_test)

