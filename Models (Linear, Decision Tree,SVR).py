#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Import all the necessary things at once
import pandas as pd
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR


# In[3]:


#take the dataset in the dataframe
df = pd.read_csv("C:\\Users\\tarun\\OneDrive\\Documents\\data\\admission_pred_data.csv")
df = df.drop('Serial No.',axis = 1)
df.describe()


# In[4]:


dd = df.loc[:,df.columns=='GRE Score']
de = df.iloc[:,-1:]


# In[5]:


#The scatter plot for all the different features
get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('GRE Score')
plt.ylabel('Chance of Admission')
plt.scatter(df.loc[:,df.columns=='GRE Score'],df.iloc[:,-1:])


# In[6]:


#The best line fit for regression model. Implement the plot


# In[7]:


#Split the training data into two sets one for training and other for test
X = df.iloc[:,:-1]
y = df.iloc[:,-1:]

#Apply the train test split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)


# In[8]:


#Linear Regression Model
lin_mod = LinearRegression()
lin_mod.fit(X_train,y_train)

#Predict the values and calculate the RMSE
y_pred = lin_mod.predict(X_test)
lin_mod_RMSE = np.sqrt(mean_squared_error(y_pred,y_test))
lin_mod_RMSE


# In[9]:


#Plot of actual values vs the predicted values
get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(y_pred,y_test)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.axis("equal")
plt.show()


# In[10]:


#Decision Tree Regressor
dec_tree_reg = DecisionTreeRegressor()
dec_tree_reg.fit(X_train,y_train)
y_dectree_pred = dec_tree_reg.predict(X_test)
dec_tree_RMSE = np.sqrt(mean_squared_error(y_dectree_pred,y_test))
dec_tree_RMSE


# In[11]:


#Plot the actual values vs the predicted values
get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(y_dectree_pred,y_test)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.axis("equal")
plt.show()
#Scatter in this model is more as compared to the Linear Regression model we can see this from the RMSE value as well


# In[17]:


#I will try to implement the SVR(Support Vector Regression)
svr_reg = SVR()
svr_reg.fit(X_train,y_train)
svr_pred = svr_reg.predict(X_test)
svr_RMSE = np.sqrt(mean_squared_error(svr_pred,y_test))
svr_RMSE


# In[18]:


#Plot the actual values vs the predicted values
get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(svr_pred,y_test)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.axis("equal")
plt.show()

