#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn


# In[13]:


dataset = pd.read_csv('datagaji.csv')

#Sumbu X adalah Pengalaman Kerja dan Sumbu Y adalah Gaji
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values


# In[24]:


from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 1/3, random_state = 0)


# In[25]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_Train, Y_Train)


# In[26]:


Y_Pred = regressor.predict(X_Test)


# In[27]:


plt.scatter(X_Train, Y_Train, color = 'red')
plt.plot(X_Train, regressor.predict(X_Train), color = 'blue')
plt.title('Gaji vs Pengalaman (Training Set)')
plt.xlabel('Pengalaman Kerja')
plt.ylabel('Gaji')
plt.show()


# In[28]:


plt.scatter(X_Test, Y_Test, color = 'red')
plt.plot(X_Train, regressor.predict(X_Train), color = 'blue')
plt.title('Gaji vs Pengalaman (Training Set)')
plt.xlabel('Pengalaman Kerja')
plt.ylabel('Gaji')
plt.show()

