# Multiple Linear Regression 

4 assumptions need to be verified before build a linear regression model : 
- Linearity
- Homoscedasticity
- Multivariate normality
- Lack of multicollinearity .   

**/!\ Dummies trap**

When we create dummy variable, we obtain a variable dependent of others.  
We fail the last assumption **"lack of multicollinearity."**  
To avoid this trap, we need always need to **remove one** of the dummy variables.   

## Dataset used 

**Subject** : Profit in function of different spend and the location for 50 companies
- independent variables : 
  - R&D spend
  - Administration spends 
  - Marketing spends 
  - State 
-  dependent variable : 
  - Profit 

## Algorithms 

### Multiple linear regression 
 
*y = b0 + b1* \* *x1 + b2* \* *x2 + ... + bn* \* *xn *  
    avec *y* = dependant variable   
         *x1, x2, ..., xn* = independent variable 
         
We can simplify the model by eliminating independent variables not significantly for the result. 
We are using for these two criteria : Adjusted R square and p value.   
As the SL has been chosen by humans, sometime we want to remove usual variables which are near the criterion.   
Adjusted R square is a good criterion to know if the variable is really important. 

#### With backpropagation elimination (with adjusted R-square)

**Initialization** : All independent variables are used<br>
**Step 1** : Fit the model with all possible predictors   
**Step 2**: Select the independent variable with the highest p_value   
**Step 3** : If this p_value is higher than the sl, this variable is not considered as significant   
        and can be removed of the model, unless the adjusted R square isn't improved by the removal        
        We return step 1.   
        Else we have reached our threshold.     
        
#### With forward selection (with adjusted R-square)

**Initialization** : Fit the model with all independent variables and choose the variable with the p-value lowest<br>
**Step 1** : Add the independent variable with the p-value lowest<br>
**Step 2** : Fit the model<br>
**Step 3** : If the p-value maximum is below the threshold, we return step 1. Else, we have reached our threshold<br>     

The model is finished with only the independent variables useful. 

## Libraries and class used 

There are two libraries which can do that in different ways

- sklearn.linear_model
  - LinearRegression : very easy to use but without the computation of pvalue and other interesting statically values
     
- statsmodels.formula.api
  - OLS : more information is computed but we need to add a constant independent variable to take in consideration the bo coefficient

## Results 

We use the quadratic normalized error to analyze models .      

Error with all independent variables take in consideration: 9137.990152794797 .   
Error after optimization by backpropagation elimination: 8198.797190788537 .   
Error after optimization by forward selection: 8198.797189010606 .   
We can see an improvement after optimization.  

