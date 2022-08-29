#!/usr/bin/env python
# coding: utf-8

# # Exercise 1
# 
# The following will be used as an in-class exercise.
# 

# 1. Write a function Square that takes an integer argument and outputs the square value of this argument. For example, if the input is 3, output should be 9.

# In[ ]:





# 2. Write a comprehensive code to initialize a dictionary where values are squares of keys – keys from 1 to 10.

# In[ ]:





# In[12]:


#Keep this here: Loading Data. 
import pandas as pd
df=pd.read_csv('https://raw.githubusercontent.com/rpi-techfundamentals/spring2019-materials/master/input/train.csv')
df


# 3. Find the median of the `age` column and assign it to the `age_median` variable. 

# In[ ]:





# 4. Count the number of NaN in the `Age` column. 
# 

# In[ ]:





# 5. Replace the NAN with the `age_median`.

# In[ ]:





# 6. Create a Pivot table which examines the survived by `Embarked` and `Sex` columns. 

# In[ ]:





# ### Challenge Problem
# 
# 6. Create a function which accepts a dataframe (`df`) and a list of columns (`cols`), and a function to use (`use` with potential values `mean` or `median` or `mode`).  For each of the columns listed the function should replace NaN with the appropriate value, returning a dataframe. 
# 
# Add your solution to this notebook:
# 
# https://colab.research.google.com/drive/1QDeA-aIjC9o2f638Hmhu_xHBQf3W1CXs?usp=sharing
# 
# Put your name. 

# In[ ]:




