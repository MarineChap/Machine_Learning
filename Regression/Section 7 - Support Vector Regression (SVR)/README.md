# Support Vector Regression (SVR) 

## Dataset used 

Subject : Salary in function of the position in the company

- independent variable :
  - Level or position in the company (1 to 10)
- dependent variable :
  - Salary

## Algorithm 

SVR with 3 kind of kernels
- Radial basis function
- Polynomial kernel
- Linear kernel

## Libraries and class used 

- sklearn.svm
  - SVR :  /!\ The library sklearn.svm doesn't handle the feature scalling. 
  
- sklearn.preprocessing
  - StandardScaler

## Results 

SVR regression is non-biased by outliers. In our case, the last point is probably considered as an outlier. 
This model is not the best idea for this particular problem but can be very useful when there are lots of outliers.

![SVR result image](https://github.com/MarineChap/Machine_Learning/blob/master/Regression/Section%207%20-%20Support%20Vector%20Regression%20(SVR)/Comparison_SVR.png)
