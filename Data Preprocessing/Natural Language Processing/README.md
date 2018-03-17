# Natural Language Processing 
    
The natural processing will transform our dataset and our problem in a classic classification problem.

## Dataset used

**Subject** :  We want to analyze reviews from a restaurant to say if it is positive or not.  <br>

**Input** : reviews with an indicator of satisfaction  <br>
After the natural processing, we are transforming the dataset like this : 
- **Independent variables** : attendance in each reviews of relevant words present in the dataset 
- **Dependent variable**    : negative or positive reviews (0/1) 

## Algorithm
In this preprocessing algorithm we want cleaning the text and create a bag world model. 

- **Step 1**: Keep only space and letters 
- **Step 2**: Transform upper letter in lower letter 
- **Step 3**: Keep only the relevant word and stem by keeping only the root of words
- **Step 4**: Create the bag of words model by making a array which list for each relevant word is the number of repetitions in each review

## Libraries and class used

- re 
  - sub : Modifiying strings with patterns 
- nltk.corpus
  - stopwords : Keep only the relevant word. Can be parametrize for different languages.
- nltk.stem.porter
  - PorterStemmer :  stem by keeping only the root of words
- sklearn.feature_extraction.text 
  - CountVectorizer : Create the bag of words model 

## Results

We do not present the classification model of the problem because it is a classic problem already studied in the classification part. 
As a result, i plot a bart diagram of the amount of use in positive and negative reviews of each relevant world. <br>
We can see the result [here](https://plot.ly/~marine_chap/24/words-present-in-restaurant-reviews/) 
