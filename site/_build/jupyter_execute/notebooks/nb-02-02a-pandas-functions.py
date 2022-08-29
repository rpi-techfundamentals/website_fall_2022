#!/usr/bin/env python
# coding: utf-8

# [![AnalyticsDojo](https://github.com/rpi-techfundamentals/spring2019-materials/blob/master/fig/final-logo.png?raw=1)](http://rpi.analyticsdojo.com)
# <center><h1>Introduction to Python - Introduction to Apply Function</h1></center>
# <center><h3><a href = 'http://rpi.analyticsdojo.com'>rpi.analyticsdojo.com</a></h3></center>
# 
# 

# # Introduction to Apply Function
# - Don't loop over a dataframe. 
# - Instead, us the apply function to process a function across each value. 
# 
# 

# In[1]:


import pandas as pd
df=pd.read_csv('https://raw.githubusercontent.com/rpi-techfundamentals/spring2019-materials/master/input/train.csv')
df


# ### Make it easy with the lambda function.
# - Create a value for `Age-squared`.

# In[2]:


df['age-squared']=df['Age'].apply(lambda x: x**2)


# ### Or define an entire function.
# - Define a function to get the title from the name. 
# - Always test your function with a single entry, not the apply.

# In[8]:


def get_title(x):
  
  x = str(x)
  x = x.split(',') #Split at the comma
  x = x[1].strip() #remove any leading spaces
  x = x.split(' ')#Split at the spaces
  return x[0]

#Always test your function with a single entry, not the apply.
get_title('Dooley, Mr. Patrick')


# In[12]:


df['Title']=df['Name'].apply(get_title)
df[['Name','Title']]


# In[13]:


df['Title'].unique()


# In[14]:


df['Title'].value_counts()


# ### Pass Additional Values
# You can even use things that pass additional values. 

# In[19]:


RECODE_MRS=['Lady.','Mme.']
RECODE_MISS=['Ms.']
RECODE_MR=['Sir.','the','Don.','Jonkheer.','Capt.']
def get_title2(x,recode_mrs, recode_miss, recode_mr):
  
  x = str(x)
  x = x.split(',') #Split at the comma
  x = x[1].strip() #remove any leading spaces
  x = x.split(' ')#Split at the spaces
  x = x[0] #select the first word. 
  if x in recode_mrs:
    x='Mrs.'
  elif x in recode_miss:
    x='Miss.'
  elif x in recode_mr:
    x='Mr.'
  return x

#Always test your function with a single entry, not the apply.
get_title('Dooley, Sir., Patrick', recode_mrs=RECODE_MRS, recode_miss=RECODE_MISS, recode_mr=RECODE_MR)


# In[20]:


df['Title']=df['Name'].apply(get_title2,recode_mrs=RECODE_MRS, recode_miss=RECODE_MISS, recode_mr=RECODE_MR )
df[['Name','Title']]


# In[21]:


df['Title'].value_counts()


# ### Using Values from more than one column
# - Apply somethign on the entire dataframe if calcs involve more than once column.

# In[24]:


def complex_process(row):
  
  return row['Sex']+str(row['Age'])

df.apply(complex_process, axis = 1)

