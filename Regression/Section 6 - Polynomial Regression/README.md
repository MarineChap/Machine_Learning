# Polynomial regression 

## Data used 

Subject : Salary in function of the position in the company

- independent variable :
  - Level or position in the company (1 to 10)
- dependent variable :
  - Salary

## Algorithm

*y = b0 + b1* \* *x + b2* \* *x\^2 + b3* \* *x\^3 + ... + bn* \* *x\^n* <br>
    avec *y* = dependent variable <br>
         *x* = independent variable <br> 
     
To compute a linear regression here, we need to transform the polynomial problem in linear problem<br>
*y = b0* \* *x0 + b1* \* *x1 + b2* \* *x2 + ... + bn* \* *xn* <br> 
         avec *y* = dependant variable<br> 
              *x1, x2, ..., xn* = independent variables<br> 
              
So, we need to transform the matrix of features to have *x0 = 1, x1 = x, x2 = x^2, ..., xn = x^n*<br> 
before applied a linear regression on the independent variables.<br> 

 The multicollinearity is still true because variables are linearly independent.<br> 
 This working in the exact same way with multiple independent variables. 
 
Step 1 : Definition of the polynomial degree
Step 2 : Transformation of the matrix of features in polynomial matrix of features
Step 3 : Linear regression with all the new independent variables

 ## Libraries and class used 
 
- sklearn.linear_model
  - LinearRegression
- sklearn.preprocessing
  - PolynomialFeatures 

 ## Result 
 
 ![Poly regression image](https://github.com/MarineChap/Machine_Learning/blob/master/Regression/Section%206%20-%20Polynomial%20Regression/Comparison_Linear_Polynomial_Regression.png)
