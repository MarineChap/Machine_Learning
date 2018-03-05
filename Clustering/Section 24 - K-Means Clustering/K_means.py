#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Part 3 : K-means cluster in following the course "Machine learning A-Z" at Udemy
    The dataset can be found here https://www.superdatascience.com/machine-learning/
    Subject : Mall want segments his clients in function of the Age, Annual Income (k$) and Spending Score (1-100)
    but it has no idea about the number or the kind of category. 

    Result can be seen here : https://plot.ly/~marine_chap/12
    
    pro : 
        - easy to understand, fast and available in lot of tools
    
    con : 
        - Important problematics with the centroid initialization 
    
Created on Wed Feb 28 17:22:19 2018
@author: marinechap

"""

# Import libraries 
import pandas as pd
from sklearn.cluster import KMeans

import plotly.plotly as py
import matplotlib.pyplot  as plt
import display_graph_online as dgo
import matplotlib.colors as cl

# Parameters 
name_file    = 'Mall_Customers.csv'
nb_indep_var =  4 

# Import dataset
dataset   = pd.read_csv(name_file)
indep_var = dataset.iloc[:,2:5].values

"""
 Elbow method for the K-means cluster
"""

wcss = []
for i in range(1,11):
    cluster =  KMeans(n_clusters = i, init = 'k-means++')
    cluster.fit(indep_var)
    wcss.append(cluster.inertia_)
    
plt.figure()
plt.plot(range(1,11), wcss, color = 'r')
plt.title('WCSS result in function of the number of clusters')
plt.xlabel('number of clusters')
plt.ylabel('WCSS')
plt.savefig('WCSS.png', bbox_inches='tight')
plt.show()

nb_cluster = 3
cluster =  KMeans(n_clusters = nb_cluster, init = 'k-means++')
k_mean = cluster.fit_predict(indep_var)
centroid = cluster.cluster_centers_ 

"""
 Display results 

""" 

data, layout = dgo.display_graph(indep_var, nb_cluster, k_mean, 'K-means clustering')
colorNames = list(cl._colors_full_map.values())

for cluster_index in range(0, nb_cluster):
        centroid_data = dict(
            mode = "markers",
            name = "centroid {0}".format(cluster_index),
            type = "scatter3d",
            marker = dict(
                    size = 10,
                    color = colorNames[cluster_index],
                    ),
            x = centroid[cluster_index, 0], y = centroid[cluster_index, 1], z = centroid[cluster_index, 2]
            )
        data.append(centroid_data)
       
fig = dict( data = data, layout = layout )
py.plot(fig, filename= 'Hierachical clustering')
