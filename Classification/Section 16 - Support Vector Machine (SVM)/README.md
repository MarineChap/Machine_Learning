# Support Vector Machine 

Assumption : 
- We don't need a dataset linearly separable <br>
The using of kernel will make the dataset linearly separable by adding a dimension

## Dataset used 

**Subject**: Predict the likelihood for a person to buy a SUV in function of his age and his estimated salary

- Independent variable :
  - Age
  - Estimed salary
- Dependent variable :
  - Purchased
  
## Algorithm 

We are using different kinds of kernels to find the hyperplan (for multidimensional problem) / line (in this case) which have the best maximum margin criterion. <br>
The maximum margin is the equal distance between the significant points of each category. These points are support vectors. <br>
There are different kernels which can be used but we will show only the linear kernel and the polynomial kernel. 


## Libraries and classes used 
- sklearn.svm
  - SVC
    - kernel parameters : 'linear', 'poly' ... etc. 
    
- sklearn.metrics
  - precision_score

## Results 

![SVM graph](https://github.com/MarineChap/Machine_Learning/blob/master/Classification/Section%2016%20-%20Support%20Vector%20Machine%20(SVM)/SVM_graph.png)
