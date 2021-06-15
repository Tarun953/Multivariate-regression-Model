#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# In[2]:


df = pd.read_csv("C:\\Users\\tarun\\OneDrive\\Documents\\data\\admission_pred_data.csv")
df.describe()


# In[7]:


X = df.loc[:,df.columns!='Serial No.']
X = X.loc[:,X.columns!='Chance of Admit']
X


# In[12]:


y = df.iloc[:,-1:]
y


# In[13]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)


# In[14]:


mod = LinearRegression()


# In[15]:


mod.fit(X_train,y_train)


# In[18]:


y_pred = mod.predict(X_test)


# In[20]:


err = np.sqrt(mean_squared_error(y_test,y_pred))
err

