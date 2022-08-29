#!/usr/bin/env python
# coding: utf-8

# #### Homework-7
# 
# This is an optional homework but you should consider it to be good practice for the Final Exam.  If you submit it will replace your 2nd lowest homework grade.  
# 
# ##### Total number of points: 40
# #### Due date: Dec 6, 2021
# 
# Before you submit this homework, make sure everything runs as expected. First, restart the kernel (in the menu, select Kernel → Restart) and then run all cells (in the menubar, select Cell → Run All). You can discuss with others regarding the homework but all work must be your own.
# 
# This homework will test your knowledge on random forests (including feature importances), and neural networks. The Python notebooks shared will be helpful to solve these problems.  
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


#Please run this cell to load the dataset. 
url = 'https://raw.githubusercontent.com/lmanikon/lmanikon.github.io/master/teaching/datasets/dataset_Facebook.csv'

import pandas as pd
import math
df  = pd.read_csv(url)
df = df.dropna()
df.columns


# #### Part-1 [24 points]: We will leverage randomforest classifier to identify the best features that predict 'Total Interactions'.

# #### 1. Create a new dataframe `df_sub2` that contains only these feature columns in `df` : 
# - `Page total likes`, `Type`, `Category`, `Post Month`, `Post Weekday`, `Post Hour`, `Paid`,  `Total Interactions`
# 
# #### 2. Transform the categorical attribute `Type` in `df_sub2` to numerical attribute this way: 
# - `Link`:1, `Photo`:2, `Status`:3, `Video`:4
# 
# #### 3. Perform Standardization (using this formula https://en.wikipedia.org/wiki/Standard_score) only on `Page total likes` column in `df_sub2` 
# - Please use `<dataframe>['<column>'].mean()` `<dataframe>['<column>'].std()` if you are using the mean and std values to manipulate the column. 
#     
# #### 4. Using `df_sub2` perform train_test_split operation to build training (`X_train`, `y_train`) and testing data (`X_test`, `y_test`).
# - Use test_size=0.3, random_state=42 as the parameters for train_test_split() function. 
# - Feature columns (X): `Page total likes`, `Type`, `Category`, `Post Month`, `Post Weekday`, `Post Hour`, `Paid`
# - CLass label column (y): `Total Interactions`
# 
# #### 5. Train the randomforest classifier (initialized as variable `rf`) using these parameters: max_depth=3, random_state=0. 
# - Using the trained model `rf` compute the impurity-based feature importances.
# - Append the names of these top-3 features to list `impFeatures`. Please make sure you type the feature names exactly as in df_sub2 
# 
# 
# 

# In[ ]:




# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


##Cell-1 -- Do not modify this cell
assert set(df_sub2.columns)=={'Post Hour', 'Total Interactions', 'Post Weekday', 'Category', 'Paid', 'Post Month', 'Page total likes', 'Type'}
assert len(df_sub2)==495
assert len(df_sub2.iloc[0,:])==8


# In[ ]:


##Cell-2 -- Do not modify this cell
assert math.ceil(df_sub2['Page total likes'].mean())==0
assert len(X_train)==346
assert 'Post Hour' in impFeatures


# In[ ]:


##Cell-3 -- Do not modify this cell
assert rf.n_classes_==230
assert len(impFeatures)==3
assert math.ceil(df_sub2['Page total likes'].std())==1


# In[ ]:


##Cell-4 -- Do not modify this cell
assert (rf.n_features_)==7
assert len(y_test)==149
assert set(impFeatures)=={'Page total likes', 'Post Hour', 'Post Weekday'}


# #### Part-2 [16 points]: We will now use neural networks to model a regression problem. 

# #### 1. Create a new dataframe `df_sub4` that contains only these feature columns in `df` : 
# - `Page total likes`, `Type`, `Category`, `Post Month`, `Post Weekday`, `Post Hour`, `Paid`,  `Total Interactions`
# 
# #### 2. Perform one-hot encoding on these features below in the dataframe `df_sub4` to create a new dataframe `df_OHE`
# - `Type`, `Category`, `Post Month`, `Post Weekday`, `Post Hour`
# 
# #### 3. Perform normalization only on `Page total likes` and `Total Interactions` column in `df_OHE` using MinMaxScaler or minmax_scale 
# - note that both these functions will output the same result
# 
# #### 4. Using `df_OHE` perform train_test_split operation to build training (`X_train`, `y_train`) and testing data (`X_test`, `y_test`).
# - Use test_size=0.3, random_state=42 as the parameters for train_test_split() function. 
# - Feature columns (X): Everything except `Total Interactions`
# - CLass label column (y): `Total Interactions`
# 
# #### 5. Build a 3-layer neural network with `relu` as the 2 hidden layers' activation function and `sigmoid` as the final layer's activation function. 
# - Use the same `adam` optimizer and `mean_squared_error` as the loss function. 
# - `epochs`=150, `batch_size`=10
# - since the class label is continuous, we use `metrics.MeanSquaredError()` as the metric (`from keras import metrics`) to measure the performance of our classifier captured as variable `mse` 

# In[ ]:


#Please run this cell to install keras
#!pip install keras
#!pip install tensorflow


# In[ ]:


# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


##Cell-5 -- Do not modify this cell
assert (math.ceil(df_OHE['Page total likes'].sum())) == 357
assert (math.ceil(df_OHE['Total Interactions'].sum())) == 17
assert (math.ceil(df_OHE['Page total likes'].mean())) == 1
assert (math.ceil(df_OHE['Total Interactions'].mean())) == 1


# In[ ]:


##Cell-6 -- Do not modify this cell
assert len(X_train)==346
assert len(y_test)==149
assert mse<0.005
assert len(model.weights)==6

