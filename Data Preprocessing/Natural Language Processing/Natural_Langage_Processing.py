#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Part 5 : Natural Langage Processing in following the course "Machine learning A-Z" at Udemy
The dataset can be found here https://www.superdatascience.com/machine-learning/
    
Objective : We want to analyze the review from a restaurant to say if it is positive or not. 
    
The natural processing will transform our dataset and our problem in a classic classification problem. 
   
For that, we will create 1 independent variable by relevant words present in the dataset 
dependent variable : negative or positive review (0/1)
  
    
Created on Fri Mar 16 15:05:08 2018
@author: marinechap

"""

# Import libraries 
import pandas as pd
import numpy as np

import re 
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

import plotly.plotly as py
import plotly.graph_objs as go

# Parameters 
name_file    = 'Restaurant_Reviews.tsv'

# Import dataset
dataset   = pd.read_csv(name_file, delimiter = '\t', quoting = 3 )
dep_value = dataset.iloc[:,1].values

# Cleaning the text
corpus = []

for index in range(0, len(dataset)):
    # Keep only space and letters 
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][index])
    # Transform upper letter in lower letter 
    review = review.lower()
    # Separate word
    review = review.split()
    
    # Keep only the relevant word and stem by keeping only the root of words
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    corpus.append(' '.join(review))
    
# Create the bag of words model 
    
cv = CountVectorizer(max_features = 1000) # To reduce the result with 1000 most frequent words in the corpus
indep_value = cv.fit_transform(corpus).toarray()
words = cv.vocabulary_

"""
Print results 
"""
positive_review = [index for index in range(0, len(dep_value)) if dep_value[index] == 1 ]
negative_review = [index for index in range(0, len(dep_value)) if dep_value[index] == 0 ]

word_in_reviews = [keys for keys in words]

positive_words = go.Bar(
            x = word_in_reviews ,
            y = [np.sum(indep_value[positive_review, words.get(keys, 'None')]) for keys in words], 
            
            name = 'Positive reviews' ,    
            marker=dict( color='green' )
    )

negative_words = go.Bar(
            x = word_in_reviews ,
            y = [np.sum(indep_value[negative_review, words.get(keys, 'None')]) for keys in words], 
            
            name = 'Negative reviews',
            marker=dict( color='red' )
    )

data = [positive_words, negative_words]

layout = go.Layout(
    barmode='stack', 
    title = 'Words present in restaurant reviews'
)

fig = go.Figure(data=data, layout=layout)

py.plot(fig, filename='Natural processing')


    