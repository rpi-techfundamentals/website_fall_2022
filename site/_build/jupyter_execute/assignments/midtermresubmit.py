#!/usr/bin/env python
# coding: utf-8

# # Midterm Resubmit
# #### Total number of points: 60
# #### Due date: Nov 11 2021 11:59  PM
# 
# This will be used to give credit back on the midterm exam. 
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


import pandas as pd

#Numerical or binary attributes
num_or_bin= ['Income', 'Kidhome','Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts','MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
       'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth','AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1','AcceptedCmp2']

#Categorical attributes
cat = ['Education','Marital_Status']

pd.set_option('display.max_columns', None)
df  = pd.read_csv("https://raw.githubusercontent.com/rpi-techfundamentals/website_fall_2021/master/site/public/marketing_campaign.csv", delimiter='\t')
df.head()


# ### Q1 [20 points]. Load and preprocess the data.  
# 
# Build and transform your dataframe `df` using the following steps before modeling. Create a new dataframe `X` that has the following properties: 
# 
# 1. It should include only the columns from lists `num_or_bin` and `cat` (in the code above).
# 2. Variables in `cat` should be transformed to `n`-1 dummy variables (where `n` is the number of categories for each variable.) 
# 3. Missing variables in the `Income` table should be filled with the median of the `income` table.
# 
# **Do not use a wrap the preprocessing in a function.  Just complete the steps asked.** 
# 

# In[ ]:


#Put your answer here. Don't create a function.  Just complete the exact steps asked. 

# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


#Please do not modify/delete this cell
# 1 Test Cell 
#this tests the columns. 
ecol = set(['Income', 'Kidhome', 'Recency', 'MntWines', 'MntFruits',
       'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
       'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
       'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth',
       'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1',
       'AcceptedCmp2', 'Education_Basic', 'Education_Graduation',
       'Education_Master', 'Education_PhD', 'Marital_Status_Alone',
       'Marital_Status_Divorced', 'Marital_Status_Married',
       'Marital_Status_Single', 'Marital_Status_Together',
       'Marital_Status_Widow', 'Marital_Status_YOLO'])
assert len(ecol-set(X.columns))==0
assert len(set(X.columns))-len(ecol)==0


# In[ ]:


#Please do not modify/delete this cell
# 2 Test Cell 
#this tests the columns. 
assert X.shape==(2240, 30)
q2=X.copy()
q2['Response']=df['Response']
q2['Year_Birth']=df['Year_Birth']
q2['Dt_Customer']=df['Dt_Customer']
q2.to_csv('q2.csv',index=False)


# In[ ]:


#Please do not modify/delete this cell
# 3 Test Cell 
assert len(X.loc[X['Income']==51381.5,'Income'])==24


# In[ ]:


#Please do not modify/delete this cell
# 4 Test Cell
assert X.isnull().sum().sum()==0


# In[ ]:


#Please do not modify/delete this cell
#Reload data
import pandas as pd
df_cl=pd.read_csv('https://raw.githubusercontent.com/rpi-techfundamentals/website_fall_2021/master/site/public/q2.csv')


# ## Q2 [20 points]. Feature Creation
# 
# 1. In the `df_cl` dataframe, create an integer feature `age` which is `2021` minus `Year_Birth`.
# 
# **NOTE: Be sure to set `age` as type `int`.**
# 
# 2. In the `df_cl` dataframe, create an integer feature `holiday` which is 1 if the month of `Dt_Customer` is equal to `12` and 0 otherwise. You can get the month by accessing `mm` from the date format dd-mm-yyyy
# 
# **NOTE: 04-09-2012 is in the form dd-mm-yyyy so for the months you want the middle numbers. Be sure to set `holiday` as type `int`.**
# 
# 
# 

# In[ ]:


#Your solution here.
# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


#Please do not modify/delete this cell
# 5 Test Cell
assert df_cl['age'].sum()==116915


# In[ ]:


#Please do not modify/delete this cell
# 6 Test Cell
import numpy as np
assert isinstance(df_cl['age'][0], (np.int64,np.int32))


# In[ ]:


#Please do not modify/delete this cell
# 7 Test Cell
assert df_cl['holiday'].sum()==175


# In[ ]:


#Please do not modify/delete this cell
# 8 Test Cell
import numpy as np
assert isinstance(df_cl['holiday'][0], (np.int64,np.int32))


# In[ ]:


#Please do not modify/delete this cell
import pandas as pd
df_cl2=pd.read_csv('https://raw.githubusercontent.com/rpi-techfundamentals/website_fall_2021/master/site/public/dfcl2.csv')
y_reload=df_cl2['Response']
X_reload=df_cl2.drop(['Response'], axis=1)


# ### Q3 [10 points] Train Test Split 
# 
# Now split the data `X_reload`, `y_reload` intro training (50%) and testing data (50%) by using these variable names while making sure there are Make sure that there are equal number of positive cases (1's) in the train and the test se 
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
# 
# **Make sure that there are an equal number of positive cases (1's) in the train and the test set!**
# 

# In[ ]:


#Put your solution here. 

# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


#Please do not modify/delete this cell
# 9 Test Cell
assert X_train.shape==(1120, 31)
assert X_test.shape==(1120, 31)


# In[ ]:


#Please do not modify/delete this cell
# 10 Test Cell
assert  y_train.sum()==y_test.sum()


# In[ ]:


#Please do not modify/delete this cell
import pandas as pd
df_cl2=pd.read_csv('https://raw.githubusercontent.com/rpi-techfundamentals/website_fall_2021/master/site/public/dfcl2.csv')
y_reload=df_cl2['Response']
X_reload=df_cl2.drop(['Response'], axis=1)

X_train2, X_test2, y_train2, y_test2 = train_test_split(X_reload, y_reload, test_size=0.3, random_state = 100)


# #### Q4 [10 points]. Classification
# 
# Use X_train2, X_test2, y_train2, y_test2 from above. Use a Gradient Boosting classifier (as instantiated below).
# 
# Calculate:
# 
# `test_acc`  The accuracy of the classifier on the test set.  
# `test_roc`  The area under the ROC curve on on the test set. 
#  
# 

# In[ ]:


from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
#use this classifier
classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)


# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


#Please do not modify/delete this cell
# 11 Test Cell
assert round(test_acc,3)==0.860


# In[ ]:


#Please do not modify/delete this cell
# 12 Test Cell
assert round(test_roc,3)==0.846

