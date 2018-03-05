# Decision Tree Regression 

## Dataset used

Subject : Salary in function of the position in the company

- independent variable :
  - Level or position in the company (1 to 10)
- dependent variable :
  - Salary

## Algorithm 

We draw a tree of value where each node is a comparator between one indepedente variable and a value. Once this done, to predict a result, we just need to follow the tree.

![Decision tree](https://github.com/MarineChap/Machine_Learning/blob/master/Regression/Section%208%20-%20Decision%20Tree%20Regression/Decision_Tree.png)

The problem is that the tree is only one random representation of the dataset. We cannot be sure to have the optimized result. 
It is why the most used algorithm is the random forest decision tree regression. 

## Libraries and class used 

- sklearn.tree 
  - DecisionTreeRegressor
  - export_graphviz : Library useful to generate the tree. Need to be installed `conda install python-graphviz`
  
- pydotplus : Library useful to generate the image of the tree. Need to be installed `pip install pydotplus`
- IPython.display : Library useful to display and save the image. 
  - Image
  - display
  
## Result 

![Decision tree result](https://github.com/MarineChap/Machine_Learning/blob/master/Regression/Section%208%20-%20Decision%20Tree%20Regression/Decision_tree_regression.png)
