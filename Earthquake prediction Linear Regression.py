#!/usr/bin/env python
# coding: utf-8

# In[232]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler as scaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


# In[233]:


data = pd.read_csv('Earthquake_data.csv')
data.head()


# In[234]:


data.describe()


# In[235]:


data.info()


# In[236]:


data.columns


# In[237]:


data.isnull().sum()


# In[238]:


#We have here deleted the string columns
data.drop(columns = ['Magt', 'SRC'],inplace = True)


# In[239]:


#Now moving to the train test split part


# In[240]:


data.head()


# In[241]:


data = data.drop(columns = ['EventID','Date(YYYY/MM/DD)','Time'] , axis=1)


# In[242]:


x = data.drop(columns='Mag',axis = 1) 
y = data['Mag']


# In[243]:


x.head()


# In[244]:


y.head()


# In[245]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state= 42)


# In[ ]:





# In[213]:


#Now scaling the data
from sklearn.preprocessing import StandardScaler

# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler to the training data and transform both training and testing data
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


# In[214]:


from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import StandardScaler

# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler to the training data and transform both training and testing data
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


# In[246]:


data.head()


# In[223]:


regressor=LinearRegression()


# In[247]:


#Fitting the model now.
regressor.fit(x_train, y_train)


# In[252]:


y_pred = regressor.predict(x_train)
y_pred


# In[259]:


#Checking the r2 scorer2= r2_score(y_test, y_pred)


# In[260]:


#Checking the length of the predictions print(len(y_test), len(y_pred))


# In[255]:



from sklearn.metrics import r2_score, mean_squared_error


# Predict on the testing set
y_pred = regressor.predict(x_test)

# Compute R^2 and MSE
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

scores['mse'].append(mse)
scores['R^2'].append(r2)

print("R^2: {:.2f}, MSE: {:.2f}".format(r2, mse))


# In[258]:


#predicting for new unseen data. Giving false data to predict for the Model
new1 = [[35.7928,-120.3353,9.88,8,89,2,0.03],[35.9463,-120.4700,12.26,7,171,20,0.02]]
new_pred = regressor.predict(new1)
new_pred


# In[ ]:




