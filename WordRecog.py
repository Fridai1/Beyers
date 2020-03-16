# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 19:09:35 2020

@author: nikolai, nikolaj, Mikkel
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# %matplotlib inline

# dataset
dataset = pd.read_csv('D:\Code\Python\Beyers\spamHam.csv')
# Data preprocessing
y = dataset.iloc[:,:1].values
X = dataset.iloc[:,1:2].values

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, random_state = 0)

y_test_raveled = y_test.ravel()
X_test_raveled = X_test.ravel()

X_train_raveled = X_train.ravel()
y_train_raveled = y_train.ravel()

# Tokenizing
cv = CountVectorizer()

# generating a wordcount as well as fitting the training data
word_count = cv.fit_transform(X_train_raveled)


# Computing the IDFs
tfidf_transformer = TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count)

# Visualizing the IDF's weights.
df_wrd = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(),columns=["idf_weights"])
df_wrd.sort_values(by=['idf_weights'])



# count matrix
count_vector=cv.transform(X_train_raveled)
count_vector

# tf-idf scores
tf_idf_vector=tfidf_transformer.transform(count_vector)
tf_idf_vector


# compute TDIDF score per document
feature_names = cv.get_feature_names()
 
# get tfidf vector for the first document
first_document_vector = tf_idf_vector[0]
first_document_vector

# print the scores
df_doc = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf"])
df_doc.sort_values(by=["tfidf"],ascending=False)

# we choose multinomial Naive Bayes 
classifier = MultinomialNB()

# connect the vectorizer to the multinomial classifier
model = make_pipeline(cv, classifier)

model.fit(X_train_raveled, y_train_raveled)

# use the trained model to predict categories for the test data
y_predicted = model.predict(X_test_raveled)



model_score = model.score(X_train_raveled, y_train_raveled)

accuracy = accuracy_score(y_test_raveled, y_predicted)
accuracy

cmat = confusion_matrix(y_test_raveled, y_predicted)
cmat

target_names_x = ['ham','spam']
target_names_y = ['spam', 'ham']

sns.set()
sns.heatmap(cmat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=target_names_x, yticklabels=target_names_y)
plt.xlabel('actual')
plt.ylabel('predicted');
plt.show()