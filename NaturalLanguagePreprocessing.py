#importing the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset (we are importing it as tab seperated values instead of csv(comma seperated values)
#so as to avoid mistakes that can occur through comma present in the reviews )

dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3) #quoting = 3 is to avoid the double quotes 

#cleaning the text
import re
import nltk

nltk.download('stopwords') #contains all the words that are irrelevent to review or identifying any type of text
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []  # a list that contains all the reviews in clean format

for i in range(0,1000):
      #we are replacing all the symbols except (a-z and A-Z) by a space
      review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
      review = review.lower()
      review = review.split()
      ps = PorterStemmer()
      #we are making use of set to make the algorithm work faster bcos it works faster in set of words instead of list
      #for each word firstly we are stemming it ie. keeping only the root word for eg. love instead of loved or loving
      # so that sparcity of matrix that will contain the column of each individual word reduces
      review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
      review = ' '.join(review)
      corpus.append(review)
      

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=1500) # we are reducing the features to 1500 ie we are taking only the 1500 most frequent words in the review

#basically creating the matrix of features
X = cv.fit_transform(corpus).toarray() #creating a huge sparse matrix by putting all the unique words in the reviews in diff column

y= dataset.iloc[:,1].values # independent variable



# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Fitting classifier to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)