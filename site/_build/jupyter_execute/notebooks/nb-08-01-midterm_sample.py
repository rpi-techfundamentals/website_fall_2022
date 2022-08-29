#!/usr/bin/env python
# coding: utf-8

# ## Sample Coding Midterm.
# 
# This is an old Midterm from a few years ago. This year will be about 16.5% harder. 
# 
# ![](https://github.com/rpi-techfundamentals/hm-01-starter/blob/master/notsaved.png?raw=1)
# 
# **WARNING!!!  If you see this icon on the top of your COLAB sesssion, your work is not saved automatically.**
# 
# 
# **Save your working file in Google drive so that all changes will be saved as you work. MAKE SURE that your final version is saved to GitHub.** 
# 
# Before you turn this in, make sure everything runs as expected. First, restart the kernel (in the menu, select Kernel → Restart) and then run all cells (in the menubar, select Cell → Run All). They should run completely without intervention...**i.e., DO NOT not manually upload any files.  Use the `wget` command to retreive files as necesssary.**
# 
# 
# ### This is a 50 point assignment.
# 
# **You may find it useful to go through the notebooks from the course materials when doing these exercises.**
# 
# **If you receive assistance from anyone in the class it it will be considered an ethical violation and referred to associate dean.**

# In[2]:


import pandas as pd
data  = pd.read_csv("https://raw.githubusercontent.com/rpi-techfundamentals/website_fall_2021/master/site/public/midterm2.csv")
data.head()


# ### (15 points) 1. Predict Group Using Different Sets of IVs 
# 
# Predict the `group` variable from `v1-v5` and then from `v6-v10` using k Nearest Neighbor and all of the data (i.e., don't do train test split) and the default hyperparameters. IGNORE THE `target` variable for now. 
# 
# `accuracy_v1_5`
# 
# `accuracy_v6_10`

# In[ ]:





# In[ ]:


accuracy_v1_5 = 


# In[ ]:


accuracy_v6_10 = 


# ### (10 points) 2. Null model
# 
# What would the accuracy of the null/naive model be?  Set it `accuracy_null`.
# 
# How would you interpret the model for `accuracy_v1_5`, `accuracy_v6_10`, vs the null model. 
# 

# In[ ]:


#Enter this to 1 decimal place. (i.e., not string)
accuracy_null= 1.1  #included as example.

accuracy_null


# In[ ]:


one_interpretation = """ 
Answer here. 
"""


# ### (15 points) 3. Perform linear regression using SciKit Learn.  
# 
# Perform two regression analyses.  
# 
# For for `analysis1` select the independent variables `v1-v10` (all v variables) and `group`.  Calculate the r2 (`r2_analysis1`) for the linear regression with the target variable.
# 
# For for `analysis2` select the independent variables `v1-v10` (all v variables) and filter out to only include `group ==1`.  Calculate the r2 `r2_analysis2` for the linear regression with the target variable.
# 
# Print `r2_analysis1` and  `r2_analysis2` to make sure they are set.
# 
# 
# 

# In[ ]:





# In[ ]:





# In[ ]:


#Print r2_analysis1 and r2_analysis2 to make sure they are set.
print(r2_analysis1, r2_analysis2)


# ### (10 points) Train Test Split
# Using the `random_state=99` do a 50 50 train test split of only variables `v1-v10` and the `target` for y.   Your split should create the following 
# 
# `train_X`, `test_X`, `train_y`, `test_y`

# In[ ]:





# In[ ]:


train_X

