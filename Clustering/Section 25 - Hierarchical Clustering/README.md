# Hierarchical clustering

## Dataset used 

**Subject** : Mall want segments his clients in function of the Age, Annual Income (k$) and Spending Score (1-100) but it has no idea about the number or the kind of category. 

- Independent variables 
  - Age
  - Annual income (K$)

- Dependent variable 
  - Spending score 
  
## Algorithm

### Divide clustering (rarely used) <br>

**Step 1** : Data starts as one combined cluster. <br>
**Step 2** : The cluster splits into two distinct parts, according to some degree of similarity. <br>
**Step 3** : Repeat step 2 until the clusters only contain a single data point. <br>  
    
### Agglomerative clustering <br> 

**Step 1** : Make each data point a single-point cluster : N clusters <br>
**Step 2** : Take the two closest clusters and make them one cluster => That forms N-1 <br>
**Step 3** : Repeat step 2 until there is only 1 cluster <br> 
    
### Common step to both algorithms : <br> 

**Step 4** : Memorization of each step in a dendrogram to choose the most efficient step of clusterization <br>
**Step 5** : Stop the clustering at the step chosen <br>
    
 
The question is **how compute** the distance between clusters ? <br>

 *Criterion of distance* : 
    - Euclidian distance 
    - Manhattan distance 
    - Cosine (cosine) distance: A good choice when there are too many variables and some variable may not be significant. Cosine distance reduces noise by taking the shape of the variables, more than their values, into account. It tends to associate observations that have the same maximum and minimum variables, regardless of their effective value.
  
  *Criterion of linkage* : 
    - Closest point (Single linkage)
    - Farest point(complete or maximum linkage)
    - Centroid 
    - Ward minimizes the variance (pooled within-group sum of squares) of the clusters being merged. 
    - Average uses the average of the distances of each observation of the two sets.<br>
            Most often used 
            
 To choose the adequate number of clusters this time, we are using the dendogram graph.<br>
 ![dendogram graph](https://github.com/MarineChap/Machine_Learning/blob/master/Clustering/Section%2025%20-%20Hierarchical%20Clustering/Dendrogram_graph.png)
    
## Libraries and classes used 

- scipy.cluster.hierarchy
- sklearn.cluster 
  - AgglomerativeClustering
- plotly.plotly : we choose for this chapter to display our result with the library plotly. The main advantage is to display the result in open-source on the web. <br>
It is one of python libraries the most promising to show the work of data scientists. 

## Result

We can see the result [here](https://plot.ly/~marine_chap/16)
