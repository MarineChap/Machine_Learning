# Naive Bayes

Why is naive ? Because of this assumption : <br>
- Variables (Age and Salary) are independent and equally important (In reality, there is probably a correlation)

## Dataset used

**Subject**: Predict the likelihood for a person to buy a SUV in function of his age and his estimated salary

- Independent variable :
  - Age
  - Estimed salary
  
- Dependent variable :
  - Purchased

## Algorithm 

This algo is based on the bayes theorem : *P(C|X) = P(X|C)* \* *P(C) / P(X)* <br>
                                                with *X* : representing the feature (observations) <br>
                                                     *C* : representing a category  <br>
Here, we have in this dataset two categories : Walker's people (W) and Driver people (D) <br>
If we want to classify a new data point : <br>
        
**Step 1** : We compute Bayes theorem for each category C. <br>
Hint : As we talked about probabilities here, you need to compute N-1 categories. <br>
(The last is equal to 1- sum(prob of other categories))<br>
- *P(C)= nbr category / nbr observation* (named prior probability)<br>
- *P(X) = nbr similar observation / nbr of observation* (named marginal likelihood)<br>
It is the probability of an observation to be in a radius around the new point<br>
=> Radius need to be fixed by static calculi.<br>
Finally, not very useful to compute this because is always identical for each variable.<br>
It doesn't affect the final comparison. <br>
- *P(X|C) = nbr similar observation of C / nbr observation of C* (named likelihood)<br>
It is the probability for a similar observation to be in the category<br>
=> Radius need to be fixed<br>

**Step 2** : Comparison of probabilities<br>
**Step 3** : Choose the category with the higher probability. 

## Libraries and classes used

- sklearn.naive_bayes
  - GaussianNB
- sklearn.metrics
  - precision_score

## Results 

![Naive bayes graph](https://github.com/MarineChap/Machine_Learning/blob/master/Classification/Section%2018%20-%20Naive%20Bayes/Naive_bayes_graph.png)
