#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Lazy r=predict library to implement the LazyRegressor
from lazypredict.Supervised import LazyRegressor


# In[2]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# In[4]:


#dataset as a dataframe
df = pd.read_csv("C:\\Users\\tarun\\OneDrive\\Documents\\data\\admission_pred_data.csv")
df = df.drop('Serial No.', axis=1)
df.describe()


# In[9]:


X = df.iloc[:,:-1]
y = df.iloc[:,-1:]


# In[10]:


#Split the training data into two sets one for training and other for test
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)


# In[11]:


reg = LazyRegressor(ignore_warnings=False, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)


# In[12]:


print(models)

