#!/usr/bin/env python
# coding: utf-8

# [![AnalyticsDojo](https://github.com/rpi-techfundamentals/spring2019-materials/blob/master/fig/final-logo.png?raw=1)](http://rpi.analyticsdojo.com)
# <center><h1>Basic Text Feature Creation in Python</h1></center>
# <center><h3><a href = 'http://rpi.analyticsdojo.com'>rpi.analyticsdojo.com</a></h3></center>

# # Basic Text Feature Creation in Python

# In[3]:


get_ipython().system('wget https://raw.githubusercontent.com/rpi-techfundamentals/spring2019-materials/master/input/train.csv')
get_ipython().system('wget https://raw.githubusercontent.com/rpi-techfundamentals/spring2019-materials/master/input/test.csv')


# In[ ]:


import numpy as np
import pandas as pd
import pandas as pd

train= pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# In[ ]:





# In[5]:


#Print to standard output, and see the results in the "log" section below after running your script
train.head()


# In[6]:


#Print to standard output, and see the results in the "log" section below after running your script
train.describe()


# In[7]:


train.dtypes


# In[8]:


#Let's look at the age field.  We can see "NaN" (which indicates missing values).s
train["Age"]


# In[9]:


#Now let's recode. 
medianAge=train["Age"].median()
print ("The Median age is:", medianAge, " years old.")
train["Age"] = train["Age"].fillna(medianAge)

#Option 2 all in one shot! 
train["Age"] = train["Age"].fillna(train["Age"].median())
train["Age"] 


# In[ ]:


#For Recoding Data, we can use what we know of selecting rows and columns
train["Embarked"] = train["Embarked"].fillna("S")
train.loc[train["Embarked"] == "S", "EmbarkedRecode"] = 0
train.loc[train["Embarked"] == "C", "EmbarkedRecode"] = 1
train.loc[train["Embarked"] == "Q", "EmbarkedRecode"] = 2


# In[ ]:


# We can also use something called a lambda function 
# You can read more about the lambda function here.
#http://www.python-course.eu/lambda.php 
gender_fn = lambda x: 0 if x == 'male' else 1
train['Gender'] = train['Sex'].map(gender_fn)


# In[12]:


#or we can do in one shot
train['NameLength'] = train['Name'].map(lambda x: len(x))
train['Age2'] = train['Age'].map(lambda x: x*x)
train


# In[ ]:



#We can start to create little small functions that will find a string.
def has_title(name):
    for s in ['Mr.', 'Mrs.', 'Miss.', 'Dr.', 'Sir.']:
        if name.find(s) >= 0:
            return True
    return False

#Now we are using that separate function in another function.  
title_fn = lambda x: 1 if has_title(x) else 0
#Finally, we call the function for name
train['Title'] = train['Name'].map(title_fn)
test['Title']= train['Name'].map(title_fn)


# In[14]:


test


# In[15]:


#Writing to File
submission=pd.DataFrame(test.loc[:,['PassengerId','Survived']])

#Any files you save will be available in the output tab below
submission.to_csv('submission.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:




