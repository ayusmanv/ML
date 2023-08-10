# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 17:58:38 2023

@author: ayusman
"""
# importing necessary modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nltk
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, ENGLISH_STOP_WORDS
from nltk import word_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier




# reading the data

df_spam = pd.read_csv('Spam Data.csv')

#print(df_spam)
#print(df.info())

# Building the vectorizer

model_vect = TfidfVectorizer(stop_words = ENGLISH_STOP_WORDS, ngram_range = (1,2))
model_vect.fit(df_spam['comments'])

X = model_vect.transform(df_spam['comments'])
# Create a DataFrame
df_transformed = pd.DataFrame(data=X.toarray(), columns=model_vect.get_feature_names_out())
#df_transformed

#Building the classifier model

y = df_spam[['spam']]
X = df_transformed

# Train test Split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state = 256 )

# Logistic Regression Based Classifier

log_reg = LogisticRegression()

log_reg.fit(X_train, y_train)


#prediction
y_pred = log_reg.predict(X_test)

#model_accuracy

print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred) / len(y_test))


# RandomForrest Based Classifier
clf = RandomForestClassifier(n_estimators= 50,random_state=256)
clf.fit(X_train, y_train)

#prediction
y_pred = clf.predict(X_test)

#model_accuracy

print("Accuracy:", accuracy_score(y_test, y_pred))

print('RandomForrest Classifier is more accurate than the Logistic Regression')



