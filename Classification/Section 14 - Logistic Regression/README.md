# Logistic regression 

It is a classification problem : We don't want to predict a result but we want a probability to be in a category. 
This algo does not work very well in case of a lot of outliers in the dataset, 
                
## Dataset used 

**Subject**: Predict the likelihood for a person to buy a SUV in function of his age 
    and his estimated salary

- Independent variable :
  - Age
  - Estimed salary

- Dependent variable :
  - Purchased  
    
## Algorithms

### Binary logistic regression (dependent variable is binary)

The probability of the answer yes (between 0 and 1) is computed by *y = 1/( 1+ exp(-x))* which is the mathematical definition of a sigmoid function.<br> 
The linear regression is still *y = b0 + b1* \* *x + b2* \* *x2* <br>
with *x1, x2* = independent variable <br>

**Step 1** : Find the parameters by linear regression <br>
**Step 2** : Applied the model on our dataset to find y <br>
**Step 3** : compute the probability p for each data <br>

### Multiple logistic regression (dependent variable has n categories)

**Step 1** : Separate the problem is n-1 binary logistic regression problem <br>
**Step 2** : Apply Binary logistic regression for each problem <br>
**Step 3** : Use the maximal probability  <br>

## Libraries and classes used 

- sklearn.linear_model 
  - LogisticRegression : 
    - Solver parameters : by default, ‘liblinear’ which is a good choice for a small dataset like this. But for Multiple logistic regression, we need to use another solver. 
  
- sklearn.metrics 
  - precision_score
  - confusion_matri

## Results 

### Confusion matrix 
The objective is to have the maximum in a and d which represent true results and the minimal in c and b which represent false results

|       |     |**estimate**|     |
|:---:  |:---:| :---:  |:---:|
|       |     |yes     |  no |
|**predict**| yes | True positive:  a = 65    | False negative: b = 3 |    
|       | no  | False positive: c = 8     | True negative: d = 24   |


### Graph 


![logistic_regression](https://github.com/MarineChap/Machine_Learning/blob/master/Classification/Section%2014%20-%20Logistic%20Regression/Logistic_regression_graph.png)
        
    
