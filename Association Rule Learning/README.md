# Apriori : Association Rules Learning

An association rule has two parts, an antecedent (if) and a consequent (then).
Association rules are created by analyzing data for frequent if/then patterns and using the criteria support, confidence and lift to identify the most important relationships.
    
*Support(M1) = User(M1) / User(dataset)*

*Confidence(M1, M2) = User(M1, M2) / User(M1)*
    
*Lift(M1, M2) = confidence (M1, M2) / support(M2)* <br>
with *M1* : antecedent <br>
     *M2* : consequent <br>

## Dataset used 

This dataset is particular. There aren't dependent or independent variables ... <br>
It is 7501 observations about a list of products by in a store by clients. The objective is to establish rules about what people buy together. 

## Algorithm

**Step 1**: Set a minimum support and confidence <br>
**Step 2**: Take all the subsets in transactions having higher support than minimum support <br>
**Step 3**: Take all the rules of theses subsets having higher confidence than minimum confidence <br>
**Step 4**: Store the rule by decreasing lift <br>

## Libraries and classes used 

- apyori  : We need to add the file in the project repository. File and download the file [here](https://pypi.python.org/pypi/apyori/1.0.0)
  - apriori

## Result 

5 first rules : 
- whole wheat pasta, mineral water       -> olive oil
- milk, mineral water, frozen vegetables -> soup
- fromage blanc 				                 -> honey
- light cream 					                 -> chicken
- pasta 						                     -> escalope
