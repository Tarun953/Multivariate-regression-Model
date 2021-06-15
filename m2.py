#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[2]:


df = pd.read_csv("C:\\Users\\tarun\\OneDrive\\Documents\\data\\dataset.csv")
df.describe()


# In[3]:


X = df.loc[:,df.columns!='Prices']
y = df.loc[:,df.columns=='Prices']


# In[4]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)


# In[5]:


mod = LinearRegression()


# In[6]:


mod.fit(X_train,y_train)


# In[7]:


y_pred = mod.predict(X_test)
y_pred


# In[8]:


np.sqrt(mean_squared_error(y_pred,y_test))

