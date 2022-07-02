#!/usr/bin/env python
# coding: utf-8

# In[39]:


#Data-Preprocessing 
# Follow following steps for data pre-processing
#1. Data Cleaning
#2. Data Integration
#3. Data Transformation
##    a.Normalization - In this method, numerical data is converted into the specified range, i.e., between 0 and one so that scaling of data can be performed.
##    b.Aggregation - The concept can be derived from the word itself, this method is used to combine the features into one. For example, combining two categories can be used to form a new group.
##    c. Generalization - In this case, lower level attributes are converted to a higher standard.
#4. Data Reduction


# In[40]:


#Importing the required lib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler


# In[44]:


#Importing the dataset
#Keep your data under input_data/ folder to read it successfully
input_matrix = pd.read_csv('C:/VAE_Project/10X/mahesh_model_imputed_10x.csv', header = 0) 
#first_column = input_matrix.columns[0]
#input_matrix = input_matrix.drop([first_column], axis=1)
#input_matrix = input_matrix.to_numpy()
print("Read data successfully")
print(input_matrix)


# In[29]:


# viewing the first few rows of the dataset
input_matrix.head()


# In[30]:


# viewing statistical info about dataset
input_matrix.describe()


# In[45]:


# Dimensionality of input matrix
ip_r = np.size(input_matrix, 0)
ip_c = np.size(input_matrix, 1)
print(ip_c)
shape = input_matrix.shape
print("Shape of input_matrix : {0}".format(shape))


# In[32]:


#Handling the missing value (NaN)
# 'np.nan' signifies that we are targeting missing values
# and the strategy we are choosing is replacing it with 'mean'
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer.fit(input_matrix.iloc[:, 1:ip_c])
input_matrix.iloc[:, 1:ip_c] = imputer.transform(input_matrix.iloc[:, 1:ip_c])  

# print the dataset
print(input_matrix)


# In[33]:


# Encoding the categorical data
#Not required in our case as we don't have any specific category in our dataset


# In[34]:


#Normalizing the dataset
'''
Feature scaling is bringing all of the features on the dataset to the same scale, 
this is necessary while training a machine learning model because in some cases the dominant features 
become so dominant that the other ordinary features are not even considered by the model.
When we normalize the dataset it brings the value of all the features between 0 and 1 so that all the columns are in the same range, 
and thus there is no dominant feature.
'''
#first_column = input_matrix.columns[0]
#input_matrix = np.delete(input_matrix, 0, 1) ##Remove first column if required
scaler = MinMaxScaler()
input_matrix.iloc[:, 1:ip_c] = pd.DataFrame(scaler.fit_transform(input_matrix.iloc[:, 1:ip_c]))
#input_matrix = pd.DataFrame(scaler.fit_transform(input_matrix.iloc[:, 0:ip_c]))
print(input_matrix)


# In[35]:


# dropping duplicate values
input_matrix.iloc[:, 1:ip_c] = input_matrix.iloc[:, 1:ip_c].drop_duplicates()
#input_matrix = input_matrix[input_matrix.iloc[:, 1:ip_c].notna()]
#input_matrix = input_matrix.drop_duplicates()
print(input_matrix.dropna())
input_matrix = input_matrix.dropna()


# In[36]:


# viewing statistical info about dataset
input_matrix.describe()


# In[37]:


# checking for missing values
input_matrix.isnull()
# checking the number of missing data
input_matrix.isnull().sum()

print(input_matrix)


# In[46]:


#Adding SUM column in last
input_matrix['sums'] = input_matrix.sum(axis=1)
input_matrix.to_csv('preprocessing_10x_sum1.csv')
shape = input_matrix.shape
print(shape)


# In[17]:


mfd_input_matrix = pd.read_csv('imputed_matrix_GSE1429_sum_filtr.csv', header = 0) 


# In[36]:


mfd_input_matrix = input_matrix
shape = input_matrix.shape
print(shape)


# In[18]:


df = pd.DataFrame(mfd_input_matrix)
mfd_input_matrix['sums'] = mfd_input_matrix.sum(axis=1)
mfd_input_matrix.drop(mfd_input_matrix[mfd_input_matrix.sums <= 400].index , inplace=True)
mfd_input_matrix.drop(mfd_input_matrix[mfd_input_matrix.sums >= 450].index , inplace=True)

print(mfd_input_matrix.shape)
mfd_input_matrix.to_csv('imputed_matrix_GSE1429_sum_filtr_f.csv')

#df = pd.DataFrame(input_matrix)
#df['sums'] = df.sum(axis=1)
#df.drop(df[df.sums <= 0.08].index , inplace=True)
#df.drop(df[df.sums >= 4.0].index , inplace=True)
#print(df)
#df.to_csv('file2.csv')


# In[ ]:





# In[ ]:




