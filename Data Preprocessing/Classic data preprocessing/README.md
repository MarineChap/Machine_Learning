# Data Preprocessing 

In this party, we see various problems meet during the preparation of a dataset and how fix them in python: 

- Import dataset and notions of independent variables and dependent variables 
- Missing data 
- Encoded categorical data and creation of dummy variables 
- Split data in a train dataset and a test dataset
- Feature scaling 

# Dataset used 

10 observations of :
- 3 independant variables (input of the model): 
  - Country : *categorical variable which creation of dummy variables*
  - Age     : *missing data + features calling*
  - Salary  : *missing data + features calling*

- 1 dependent variable (output of the model): 
  - Purchased : *categorical variable*

# Libraries and class used 

- pandas : *importation dataset*
- sklearn.preprocessing
  - Imputer        : *missing values*
  - LabelEncoder   : *categorical variables*
  - OneHotEncoder  : *creation of dummy variables*
  - StandardScaler : *features calling*
  
- sklearn.model_selection  
  - train_test_split : *split dataset*
