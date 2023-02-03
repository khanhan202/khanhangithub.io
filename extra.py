#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import metrics 
from sklearn.decomposition import PCA


# In[8]:


def evaluate(clf, X_data, y_data):
    accuracy = cross_val_score(clf, X_data, y_data, cv=10, scoring = 'accuracy')
    print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))

    precision = cross_val_score(clf, X_data, y_data, cv=10, scoring = 'precision')
    print("Precision: %0.5f (+/- %0.5f)" % (precision.mean(), precision.std() * 2))

    recall = cross_val_score(clf, X_data, y_data, cv=10, scoring = 'recall')
    print("Recall: %0.5f (+/- %0.5f)" % (recall.mean(), recall.std() * 2))

    f = cross_val_score(clf, X_data, y_data, cv=10, scoring = 'f1')
    print("F-measure: %0.5f (+/- %0.5f)" % (f.mean(), f.std() * 2))


# In[9]:


def predict(clf, X_test, y_test):
    print("------------------Testing Result-------------------)")
    y_pred = clf.predict(X_test)
    ac1 = accuracy_score(y_pred, y_test)*100
    print("Accuracy is: ", ac1)
    print()
    print(classification_report(y_test, y_pred, digits=5))


# In[10]:


minmax_scale = MinMaxScaler(feature_range=(0, 1))
def normalization(df,col):
  for i in col:
    arr = df[i]
    arr = np.array(arr)
    df[i] = minmax_scale.fit_transform(arr.reshape(len(arr),1))
  return df


# In[1]:


def my_PCA(X, percent):
    n_componets = round(percent * len(X.columns))
    pca = PCA(n_componets)
    pca.fit(X)
    new_X = pca.transform(X)
    return new_X


# In[ ]:




