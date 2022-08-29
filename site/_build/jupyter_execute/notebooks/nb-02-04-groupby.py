#!/usr/bin/env python
# coding: utf-8

# [![AnalyticsDojo](https://github.com/rpi-techfundamentals/spring2019-materials/blob/master/fig/final-logo.png?raw=1)](http://introml.analyticsdojo.com)
# <center><h1>Introduction to Python - Groupby and Pivot Tables</h1></center>
# <center><h3><a href = 'http://introml.analyticsdojo.com'>introml.analyticsdojo.com</a></h3></center>
# 
# 
# 

# # Groupby and Pivot Tables

# In[1]:


import numpy as np 
import pandas as pd 

# Input data files are available in the "../input/" directory.
# Let's input them into a Pandas DataFrame
train = pd.read_csv('https://raw.githubusercontent.com/rpi-techfundamentals/spring2019-materials/master/input/train.csv')
test  = pd.read_csv('https://raw.githubusercontent.com/rpi-techfundamentals/spring2019-materials/master/input/test.csv')


# ### Groupby
# - Often it is useful to see statistics by different classes.
# - Can be used to examine different subpopulations

# In[2]:


train.head()


# In[3]:


print(train.dtypes)


# In[4]:


#What does this tell us?  
train.groupby(['Sex']).Survived.mean()


# In[5]:


#What does this tell us?  
train.groupby(['Sex','Pclass']).Survived.mean()


# In[6]:


#What does this tell us?  Here it doesn't look so clear. We could separate by set age ranges.
train.groupby(['Sex','Age']).Survived.mean()


# ### Combining Multiple Operations
# - *Splitting* the data into groups based on some criteria
# - *Applying* a function to each group independently
# - *Combining* the results into a data structure

# In[7]:


s = train.groupby(['Sex','Pclass'], as_index=False).Survived.sum()
s['PerSurv'] = train.groupby(['Sex','Pclass'], as_index=False).Survived.mean().Survived
s['PerSurv']=s['PerSurv']*100
s['Count'] = train.groupby(['Sex','Pclass'], as_index=False).Survived.count().Survived
survived =s.Survived
s


# In[8]:


#What does this tell us?  
spmean=train.groupby(['Sex','Pclass']).Survived.mean()
spcount=train.groupby(['Sex','Pclass']).Survived.sum()
spsum=train.groupby(['Sex','Pclass']).Survived.count()
spsum


# ### Pivot Tables
# - A pivot table is a data summarization tool, much easier than the syntax of groupBy. 
# - It can be used to that sum, sort, averge, count, over a pandas dataframe. 
# - Download and open data in excel to appreciate the ways that you can use Pivot Tables. 

# In[10]:


#List the index and the functions you want to aggregage by. 
pd.pivot_table(train,index=["Sex","Pclass"],values=["Survived"],aggfunc=['count','sum','mean'])


# In[ ]:




