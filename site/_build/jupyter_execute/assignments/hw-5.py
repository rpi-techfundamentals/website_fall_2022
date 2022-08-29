#!/usr/bin/env python
# coding: utf-8

# ###### Homework-5
# #### Total number of points: 70
# #### Due date: Oct 21th 2021
# 
# Before you submit this homework, make sure everything runs as expected. First, restart the kernel (in the menu, select Kernel → Restart) and then run all cells (in the menubar, select Cell → Run All). You can discuss with others regarding the homework but all work must be your own.
# 
# This homework will test your knowledge on data manipulation, manipulating strings and feature preprocessing. The Python notebooks shared will be helpful to solve these problems.  
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


#This is a fix for some of the tests. 
from math import log10, floor
def round_sig(x, sig=2):
    return round(x, sig-int(floor(log10(abs(x))))-1)


# ### Q1 [10 points]. Folds for cross validation. 
# 
# Use the [KFold()](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html) function from scikit-learn to generate 4 folds for cross validation.  
# 
# For each fold -- append the mean of training data to `traindata_means`
# and the mean of testing data to `testdata_means` 
# 
# #Hint -- Use np.mean() to compute the mean values.
# 
# 

# In[ ]:



from sklearn.model_selection import KFold
import numpy as np
l1=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
traindata=[]
testdata=[]

# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


#Please do not modify/delete this cell
assert np.mean(traindata)==65.0
assert len(testdata)==4


# In[ ]:


#Please do not modify/delete this cell


# In[ ]:


#Please do not modify/delete this cell
import pandas as pd
df_boston_train=pd.read_csv('https://raw.githubusercontent.com/rpi-techfundamentals/website_fall_2021/master/site/public/boston_train.csv')


# ### Q2 [10 points]. Boston Preprocessing
# 
# Create a function `preprocess` which accepts a dataframe (`df`), a list of numeric columns (`num`), a list of columns that are categorical (`cat`), and optionionally the dependent variable (`dv`). 
# 
# 1. For categorical columns, the function should replace all missing values with their own category 'missing' (setting NAs equal to the string 'missing'). It should then generate n-1 dummy variables for each categorical variables. 
# 
# 2. For numeric columnns, the function should replace all missing values with the median.
# 
# 3. The function should also check whether the `dv` is in the dataframe. If it is in the dataframe return `X` and `y`.  If the dv is not in the dataframe, return just `X`.
# 
# Hint: [This notebook](https://introml.analyticsdojo.com/notebooks/nb-04-06-revisit-titanic.html) has a solid starting point for you to inspect. 
# 

# In[ ]:


#Please do not modify/delete this cell
#This selects the numeric and categorical columns.
num_features = list(df_boston_train.select_dtypes(include=[np.number]).columns.values)
cat_features = list(set(df_boston_train.columns)-set(num_features))
num_features.remove('Id') #Why do we drop Id? You should know answer for midterm.
num_features.remove('SalePrice')
dv='SalePrice'


# In[ ]:



#Your solution here.

# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


#Please do not modify/delete this cell
X,y=preprocess(df_boston_train, num_features, cat_features, dv)
X2=preprocess(df_boston_train, num_features, cat_features)

assert X.shape[0]==1460
assert X.shape[1]==261
assert y.sum()==264144946


# In[ ]:





# ### Q3 [10 points] Regression Function
# 
# Don't worry about train/test split our anything.  Just run a regression using scikit learn. 
# 
# Create `y_pred` as the predicted value from the regression model using `X_reload` and `y_reload`. 
# 
# Calculate the overall R2 (`reg_r2`) for the model as well as the mean squared error (`reg_mse`). 
# 

# In[ ]:


#Please do not modify/delete this cell
#We are loading processed data. Don't change this.  
X_reload = pd.read_csv('https://raw.githubusercontent.com/rpi-techfundamentals/website_fall_2021/master/site/public/processed_housing.csv')
y_reload = X_reload['SalePrice']
X_reload = X_reload.drop('SalePrice', axis=1)


# In[ ]:



# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


#Please do not modify/delete this cell
assert round(reg_r2,3)==0.931
assert round_sig(reg_mse)==round_sig(438286412)


# In[ ]:





# #### Q4 [10 points]. Train Test Split
# 
# Now split the data `X_reload`, `y_reload` intro training (80%) and testing data (20%) by using these variable names 
# 
# `X_train`: Training feature columns
# 
# `X_test`: Testing feature columns
# 
# `y_train`: Training DV
# 
# `y_test`: Testing DV
# 
# When doing so set the `random_state` parameter to `100` so you get the same split that is in the solution.

# In[ ]:


# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


assert len(X_train)==1168
assert len(y_test)==292


# In[ ]:


#Please do not modify/delete this cell


# #### Q5 [10 points]. Train Test Split for Regression Training
# 
# Create a function `train_regression` which accepts `X_train`, `X_test`, `y_train`, `y_test` and returns the following:
# 
# 
# 1. `y_pred_train` and `y_pred_test`as the predicted values from the regression models.
# 
# 2. The R2 for the training (`reg_r2_train`) and test data (`reg_r2_test`).
# 
# 3. The mean squared error for the training (`reg_mse_train`) and test data (`reg_mse_test`).
# 

# In[ ]:


def train_regression(X_train, X_test, y_train, y_test):
    #Your Code here
    return y_pred_train, y_pred_test, reg_r2_train, reg_r2_test, reg_mse_train, reg_mse_test
    

# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


assert round(reg_r2_train,3)==0.943
assert reg_r2_test <0
assert round_sig(reg_mse_train)==round_sig(358143388)


# In[ ]:


#Please do not modify/delete this cell


# #### Q6 [10 points]. PCA
# 
# Perform PCA on your `X_reload` data, with parameter such that you extract 90% of the variance in your independent variables, to create `X_pca` the `X_pca`.  Use `random_state`=100. 
# 
# Then repeat the steps of splitting the data into `X_train_pca`, `X_test_pca`, `y_train_pca`, `y_test_pca` and run the final analysis on your regression function `train_regression`. 
# 
# 

# In[ ]:


# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


assert X_pca.shape[1]==5
assert round(reg_r2_train,3) ==0.622
assert round_sig(reg_mse_train)== round_sig(2369212459)


# In[ ]:


#Please do not modify/delete this cell


# #### Q7 [10 points]. Scaling Model Serach. 
# 
# So what you should have found in the above analysis was that a standard linear regression model when fed data many columns of data that are highly correlated does worse than just a simple model build on a few principal components. 
# 
# **However, is there important information in the data which other algorthims might be able utilize to make better preditctions?**
# 
# Create a funtion `regmodel` which accepts a model dictionary of however many Scikitlearn regression models.  
# 
# ```
# def regmodel(model_dictionary, X_train, X_test, y_train, y_test, final_order):
#     #*****your code****
#     return results_df
# 
# ```
# It should run each model in the dictionary and return a dataframe summarizing the results, as follows:
# 
# ![image.png](attachment:image.png)
# 
# *Hint: [This notebook](https://introml.analyticsdojo.com/notebooks/nb-04-06-revisit-titanic.html) has a solid starting point for you to inspect.* 
# 
# 

# In[ ]:


import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor

#These are the parameters you pass to your model.
#setting randomstate so you get the same results as i did. 
allmodels={"Random Forest": RandomForestRegressor( random_state=100),
           "AdaBoost": AdaBoostRegressor( random_state=100)}
final_order=['name','r2-train', 'r2-test', 'mse-train', 'mse-test']


# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


#Please do not modify/delete this cell
#Half credit for getting columns of resulting dataframe correct
results_df = regmodel(allmodels, X_train, X_test, y_train, y_test, final_order)
assert set(results_df.columns)==set(['name','r2-train', 'r2-test', 'mse-train', 'mse-test'])


# In[ ]:


#Please do not modify/delete this cell
assert round(results_df['r2-train'][0],3)==0.976
assert round_sig(results_df['mse-train'][0])== round_sig(147830303)
assert round(results_df['r2-test'][0],3)==0.876
assert round_sig(results_df['mse-test'][0])== round_sig(795442670)


# 
# ### Optional Future Enhancements
# 
# 1. Add K-Fold cross validation form  and include the results of 5 or 10 different models. Just adding an inner loop. 
# 
# 2. Try different models.  With your function, you could evaluate the effect of any number of regression models. 
# 
# [https://scikit-learn.org/stable/supervised_learning.html](https://scikit-learn.org/stable/supervised_learning.html)

# In[ ]:




