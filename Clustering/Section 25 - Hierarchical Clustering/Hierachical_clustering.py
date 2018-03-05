#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Part 3 : Hierachical cluster in following the course "Machine learning A-Z" at Udemy
    The dataset can be found here https://www.superdatascience.com/machine-learning/
    Subject : Mall want segments his clients in function of the Age, Annual Income (k$) and Spending Score (1-100)
    but it has no idea about the number or the kind of category. 

Divise clustering (rarely used)

    Step 1 : Data starts as one combined cluster.
    Step 2 : The cluster splits into two distinct parts, according to some degree of similarity.
    Step 3 : Repeat step 2 until the clusters only contain a single data point.  
    
Agglomerative clustering 
    Step 1 : Make each data point a single-point cluster : N clusters
    Step 2 : Take the two closest cluster and make them one cluster => That forms N-1
    Step 3 : Repeat step 2 until there are only 1 cluster 
    
Common step to the both algorithms : 
    Step 4 : Memorization of each step in a dendrograms to choose the most efficient step of clusterization
    Step 5 : Stop the clustering at the step choosen
    Result can be seen here : https://plot.ly/~marine_chap/16
    
Created on Sat Mar  3 18:47:31 2018
@author: marinechap

"""

# Import libraries 
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering 

import matplotlib.pyplot  as plt
import display_graph_online as dgo
import plotly.plotly as py

# Parameters 
name_file    = 'Mall_Customers.csv'
nb_indep_var =  4 

# Import dataset
dataset   = pd.read_csv(name_file)
indep_var = dataset.iloc[:,2:5].values

"""
 Using the dendogram graph to choose the number of clusters
 
 The question is how compute the distance between clusters ? 
  Criterion of distance : 
    - Euclidian distance 
    - Manhattan distance 
    - Cosine (cosine) distance: A good choice when there are too many variables 
    and some variable may not be significant. Cosine distance reduces noise 
    by taking the shape of the variables, more than their values, into account. 
    It tends to associate observations that have the same maximum and minimum variables,
    regardless of their effective value.
  
  Criterion of linkage : 
    - Closest point (Single linkage)
    - Farest point(complete or maximum linkage)
    - Centroid 
    - Ward minimizes the variance (pooled within-group sum of squares) of the clusters being merged. 
    - Average uses the average of the distances of each observation of the two sets.
            Most often used 

        
"""

sch.dendrogram(sch.linkage(y= indep_var, method = 'average'))
plt.title('Dendrogram graph')
plt.xlabel('Costumers')
plt.ylabel('Euclidian distance')
plt.savefig('Dendrogram_graph.png', bbox_inches='tight')
plt.show()


# Fit the model Hierachical clustering 

nb_cluster = 5
cluster = AgglomerativeClustering(n_clusters = nb_cluster, linkage = 'average')

HC = cluster.fit_predict(indep_var)

"""
 Display results 

""" 

data, layout = dgo.display_graph(indep_var, nb_cluster, HC, 'Hierachical clustering')
fig = dict( data = data, layout = layout )
py.plot(fig, filename= 'Hierachical clustering')