# K-Nearest Neighbors 

## Dataset used
Subject: Predict the likelihood for a person to buy a SUV in function of his age and his estimated salary

- Independent variable :
  - Age
  - Estimed salary
- Dependent variable :
  - Purchased
  
## Algorithm 

**Step 1** : Choose K nearest neighbors (usually 5) <br>
**Step 2** : Take the k-NN of the new data point, according to the Euclidian distance (or Manhattan distance) <br>
**Step 3** : Count the number of NN in each category<br>
**Conclusion** : The new data point is in the category the most represented in his neighborhood.<br>

## Libraries and classes used 
- sklearn.neighbors
  - KNeighborsClassifier
  
- sklearn.metrics
  - precision_score
  - confusion_matrix

## Results 
### Confusion matrix

|       |     |**estimate**|     |
|:---:  |:---:| :---:  |:---:|
|       |     |yes     |  no |
|**predict**| yes | True positive:  a = 64    | False negative: b = 4 |    
|       | no  | False positive: c = 3     | True negative: d = 29   |


### Graph
![K-NN image](https://github.com/MarineChap/Machine_Learning/blob/master/Classification/Section%2015%20-%20K-Nearest%20Neighbors%20(K-NN)/K_NN_graph.png)
