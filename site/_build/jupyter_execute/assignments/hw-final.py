#!/usr/bin/env python
# coding: utf-8

# ## Final Exam
# #### 81 Points
# #### Due date: Dec 17 7:30 PM
# 
# Before you submit this homework, make sure everything runs as expected. First, restart the kernel (in the menu, select Kernel → Restart) and then run all cells (in the menubar, select Cell → Run All). You can discuss with others regarding the homework but all work must be your own.
# 
# Steps to evaluate your solutions:
# 
# Step-1: Try on Colab or Anaconda (Windows: https://docs.anaconda.com/anaconda/install/windows/ ; Mac:https://docs.anaconda.com/anaconda/install/mac-os/ ; Linux: https://docs.anaconda.com/anaconda/install/linux/)
# 
# Step-2: Open the Jupyter Notebook by first launching the anaconda software console
# 
# Step-3: Open the homework's .ipynb file and write your solutions at the appropriate location "# YOUR CODE HERE"
# 
# Step-4: You can restart the kernel and click run all (in the menubar, select Cell → Run All) on the center-right on the top of this window.
# 
# Step-5: Now go to "File" then click on "Download as" then click on "Notebook (.ipynb)" Please DO NOT change the file name and just keep it as ".ipynb"
# 
# Step-6: Go to lms.rpi.edu and upload your homework at the appropriate link to submit this homework.
# 
# 
# #### Please note that for any question in this assignment you will receive points ONLY if your solution passes all the test cases including hidden testcases as well. So please make sure you try to think all possible scenarios before submitting your answers.  
# - Note that hidden tests are present to ensure you are not hardcoding. 
# - If caught cheating: 
#     - you will receive a score of 0 for the 1st violation. 
#     - for repeated incidents, you will receive an automatic 'F' grade and will be reported to the dean of Lally School of Management. 

# In[ ]:


#Please do not modify/delete this cell
import pandas as pd
pd.set_option('display.max_columns', None)
df  = pd.read_csv("https://raw.githubusercontent.com/rpi-techfundamentals/website_fall_2021/master/site/public/data2.csv")
df


# ### Q1 [20 points]. Cluster Analysis  
# 
# For the cluster analysis below use all variables in `df` except the `target` column. 
# 
# 1. Using the KMEANS function to create a `kmeans` cluster mode in sklearn. Create columns in df called `kmeans_3` and `kmeans_5` which includes the cluster `labels` for a cluster analysis with `3` and `5` clusters. Use `random_state=100` as a parameter in the KMeans. 
# 
# 
# 2. Create a dictionary from the analysis in 1 called `inert` in which the key of the dictionary is the number of clusters (3 and 5 in `int` type) and the value is the sum of squared distances of samples to their closest cluster center (i.e., the inertia). 
# 
# #For example
# ```
# inter[3] = <get the inertia for the model with 3 clusters>
# inter[5] = <get the intertia for the model with 5 clusters>
# 
# ```
# 
# 

# In[ ]:


#Put your entire answer here in this cell. 
#It is ok to use other temporary cells while working but for final submission move all related code to this cell.

# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


#These are helper tests to make sure you have answers.  If this fails, please check your naming. 
assert 'kmeans_3' in df.columns
assert 'kmeans_5' in df.columns
assert 'inert' in globals()


# In[ ]:


#Please do not modify/delete this cell


# In[ ]:


#Please do not modify/delete this cell


# In[ ]:


#Please do not modify/delete this cell


# In[ ]:


#Please do not modify/delete this cell


# In[ ]:


#Please do not modify/delete this cell
#Reload data
import pandas as pd
from sklearn.model_selection import train_test_split
df_q2= pd.read_csv("https://raw.githubusercontent.com/rpi-techfundamentals/website_fall_2021/master/site/public/data2.csv")
train_X, test_X, train_y, test_y = train_test_split(df_q2.iloc[:,1:], df_q2['target'], 
                                                    train_size=0.7,
                                                    test_size=0.3,
                                                    random_state=122)


# ## Q2 [21 points]. GBM Regressor
# 
# Using the training and test data provided (`train_X`, `test_X`, `train_y`, `test_y`), use the `sklearn.ensemble.GradientBoostingRegressor` with `random_state = 100`.
# 
# 1. Calculate the r-squared for both the `train` and the `test` set and assign to `gbm_r2_train` and `gbm_r2_test`, respectively.
# 
# 2. Calculate the mean absolute error for both the `train` and the `test` set and assign to `gbm_mae_train` and `gbm_mae_test`, respectively.
# 
# 3. Create a varaible `gbm_most_important` and indicate the column name of the most important column from the GradientBoostingRegressor analysis. 
# 

# In[ ]:


#Put your entire answer here in this cell. 
#It is ok to use other temporary cells while working but for final submission move all related code to this cell.

# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


#These are helper tests to make sure you have answers.  If this fails, please check your naming. 
assert 'gbm_mae_train' in globals()
assert 'gbm_mae_test' in globals()
assert 'gbm_r2_test' in globals()
assert 'gbm_r2_train' in globals()
assert 'gbm_most_important' in globals()


# In[ ]:


#Please do not modify/delete this cell


# In[ ]:


#Please do not modify/delete this cell


# In[ ]:


#Please do not modify/delete this cell


# In[ ]:


#Please do not modify/delete this cell


# In[ ]:


#Please do not modify/delete this cell


# ### Q3 [15 points] Regression
# Using the  `df_q2` dataframe, use the `sklearn.linear_model.LinearRegression` to do linear regression.
# 
# 1. Calculate the r-squared for both the `train` and the `test` set and assign to `reg_r2_train` and `reg_r2_test`, respectively.
# 
# 2. Calculate the mean absolute error for both the `train` and the `test` set and assign to `reg_mae_train` and `reg_mae_test`, respectively.
# 
# 3. From you regression model, get the  beta coefficient of the column `treatment` and set it to the variable  `treat_coef`. 
# 
# 4. Based on the coefficient (don't worry about significance) does the treatment tend to `increase` or `decrease` the `target`? Set the value of `treat_dir` to your answer. 
# 
# 
# 
# 

# In[ ]:


#Put your entire answer here in this cell. 
#It is ok to use other temporary cells while working but for final submission move all related code to this cell.


# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


#These are helper tests to make sure you have answers.  If this fails, please check your naming. 
assert 'reg_mae_train' in globals()
assert 'reg_mae_test' in globals()
assert 'reg_r2_test' in globals()
assert 'reg_r2_train' in globals()
assert 'treat_coef' in globals()
assert treat_dir in ['increase','decrease']


# In[ ]:


#Please do not modify/delete this cell


# In[ ]:


#Please do not modify/delete this cell


# In[ ]:


#Please do not modify/delete this cell


# In[ ]:


#Please do not modify/delete this cell


# #### Q4[15 points]. Deep Learning. 
# 
# Build a 4-layer neural network called `model`:
# 
# - Layer 0 is a normalization layer.
# - Layer 1 is a dense layer with 100 nodes and relu activation
# - Layer 2 is a dense layer with 100 nodes and relu activation
# - Layer 3 is the final dense layer with 1 output. 
# 
# 
# Use the `adam` optimizer and `mean_absolute_error` as the loss function with `epochs`=175.
# 
# Also, evaluate the model. Calculate `deep_r2_train`, `deep_mae_train`,  `deep_r2_test`, and   `deep_mae_test`.  
# 
# 

# In[ ]:


#Put your entire answer here in this cell. 
#It is ok to use other temporary cells while working but for final submission move all related code to this cell.


# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


assert 'model' in globals()
assert 'deep_r2_train' in globals()
assert 'deep_mae_train' in globals()
assert 'deep_r2_test' in globals()
assert 'deep_mae_test' in globals()


# In[ ]:


#Please do not modify/delete this cell


# In[ ]:


#Please do not modify/delete this cell


# In[ ]:


#Please do not modify/delete this cell


# In[ ]:


#Please do not modify/delete this cell


# #### Q5 [17.5 points]. Comparison
# 
# For each of the questions below, reply with the available options in `()`. Even if you didn't get all models to work, at least make a guess. Hint: The intuition of what you might expect for the 3 different models is correct. 
# 
# `q5a`  Which is a linear model? (`gbm` or `reg` or `deep`)
# 
# `q5b`  Which is actually make up of many different models working together (an ensemble model) (`gbm` or `reg` or `deep`)?
# 
# `q5c`  Which performs better from a prediction standpoint (`gbm` or `reg` or `deep`)? 
# 
# `q5d`  Which gives you an understanding of the direction of the treatment on the target (`gbm` or `reg` or `deep`)?
# 
# `q5e`  Which shows more evidence of overfitting between train and test (`gbm` or `reg` or `deep`)? 
# 
# 
# `q5f`  Which model has the most parameters?(`gbm` or `reg` or `deep`)
# 
# `q5g`  The importance metric from a GBM analysis is always very highly correlated with the coefficients from a regression analysis (`True` or `False`).
#  
# 

# In[ ]:


#Put your entire answer here in this cell. 
#It is ok to use other temporary cells while working but for final submission move all related code to this cell.


# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


#These are helper tests to make sure you have answers.  If this fails, please check your naming. 
assert q5a in ['reg','gbm','deep']
assert q5b in ['reg','gbm','deep']
assert q5c in ['reg','gbm','deep']
assert q5d in ['reg','gbm','deep']
assert q5e in ['reg','gbm','deep']
assert q5f in ['reg','gbm','deep']
assert q5g in ['True','False']


# In[ ]:


#Please do not modify/delete this cell


# In[ ]:


#Please do not modify/delete this cell


# In[ ]:


#Please do not modify/delete this cell


# In[ ]:


#Please do not modify/delete this cell


# In[ ]:


#Please do not modify/delete this cell


# In[ ]:


#Please do not modify/delete this cell


# In[ ]:


#Please do not modify/delete this cell

