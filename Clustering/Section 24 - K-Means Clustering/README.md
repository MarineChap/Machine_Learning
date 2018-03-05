# K-means clustering

Clustering is a different problem than classification. In this case, we don't have any idea of patterns and how separate data. 
So, before, we need to do some tests for knowing how many clusters we need to implement. 

## Dataset used 

**Subject** : Mall want segments his clients in function of the Age, Annual Income (k$) and Spending Score (1-100) but it has no idea about the number or the kind of category. 

- Independent variables 
  - Age
  - Annual income (K$)

- Dependent variable 
  - Spending score 
  
## Algorithm

**Step 1**: Choose K random points as centroid <br>
**Step 2**: Associate each point for each centroid <br>
**Step 3**: Replace the centroid and return to the step 2 until there is no change. <br>

/!\ Random Initialization Trap <br>
Following how the random initialization falls, the algorithm won't give the same result or converge in the same way. <br>
The most part of libraries take this problem in charge but we still need to be aware. <br>

Choose the right number of clusters is also a important parameter. For this, we are using the WCSS metrics for "The Elbow method". <br>

![WCSS](https://github.com/MarineChap/Machine_Learning/blob/master/Clustering/Section%2024%20-%20K-Means%20Clustering/WCSS.png)

The WCSS metric is the sum of the distance between each cluster and each point square. The adding of a cluster decrease the metric but at a certain moment, the decrease is not important enough to be justified. We choose here 3 clusters. 

## Libraries and classes used 

- sklearn.cluster
  - KMeans
- plotly.plotly : we choose for this chapter to display our result with the library plotly. The main advantage is to display the result in open-source on the web. <br>
It is one of python libraries the most promising to show the work of data scientists. 

## Result

We can see the result [here](https://plot.ly/~marine_chap/18/k-means-clustering/)



