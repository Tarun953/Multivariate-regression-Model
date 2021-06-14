#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("C:/Users/tarun/OneDrive/Documents/data/dataset.csv")
df


# In[3]:


df.head()


# In[4]:


df.describe()


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('area(Sq. feet)')
plt.ylabel('Price (US$)')
plt.scatter(df.Area,df.Prices)


# In[6]:


X = df[['Area','Garage','FirePlace','Baths','White Marble','Black Marble','Indian Marble','Floors','City','Solar','Electric','Fiber','Glass Doors','Swiming Pool','Garden']]
y = df['Prices']


# In[7]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
len(X_test)


# In[8]:


reg = linear_model.LinearRegression()
reg.fit(X_train,y_train)


# In[9]:


reg.predict(X_test)


# In[10]:


reg.score(X_test,y_test)

