#!/usr/bin/env python
# coding: utf-8

# ##### Homework-4
# ##### Total number of points: 50
# #### Due date: October 7, 2021
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

# #### Q1 [10 points]. Given the groundtruth or actual class labels `y_true` and the predicted class labels `y_pred`, 
# #### build a confusion matrix without using any existing library to compute 
# - `fp` false positives, 
# - `tp` true positives, 
# - `fn` false negatives and 
# - `tn` true negatives. 
# 

# In[ ]:



y_true=[1, 1, 0, 0, 1, 1, 1, 1]
y_pred=[1, 0, 0, 1, 1, 0, 1, 1]

tp=0
fp=0
fn=0
tn=0

# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


#Please do not modify/delete this cell
assert tp==4
assert fp==1


# In[ ]:


#Please do not modify/delete this cell


# #### Q2 [5 points]. Given the `y_true` which is the groundtruth (or the actual class labels), create a list of predicted labels `y_pred` 
# #### that reflects the respective worst-performing model where accuracy will be '0'. 
# 
# 

# In[ ]:


from sklearn.metrics import accuracy_score

y_true = [1, 1, 1, 1, 1, 1, 1, 1]

# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


#Please do not modify/delete this cell
assert accuracy_score(y_true, y_pred)==0


# #### Q3 [10 points]. More Calculations of Classification Performance 
# 
# By utilizing the proposed `y_prob` and `y_true` and using a threshold of 0.5 (where the class is 1 if the probability is >= 0.5 and 0 if the probability is < 0.5) calculate the predicted class `y_pred`. Compute the accuracy as `acc`, true positive rate as `tpr`, sensitivity value as `sens`, and Area Under the Receiver Operating Characteristic Curve (ROC AUC) as `auc`. 
# 
# **Hint, one of the evaluation functions takes the probability as an input.**
# 

# In[ ]:


y_true=[0, 1, 0, 0, 1, 1, 1, 0]
y_prob=[0.40, 0.70, 0.20, 0.30, 0.40, 0.65, 0.70, 0.80]


# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


#Please do not modify/delete this cell
assert y_pred == [0, 1, 0, 0, 0, 1, 1, 1]
assert round(acc,3)==0.75
assert round(sens,3)==0.75
assert round(tpr,3)==0.75


# In[ ]:


#Please do not modify/delete this cell
assert round(auc,3)== 0.719


# In[ ]:


#Please do not modify/delete this cell
import pandas as pd
df=pd.read_csv('https://raw.githubusercontent.com/rpi-techfundamentals/website_fall_2021/master/site/public/titanic_processed_train3.csv')
#View the dataframe
df


# ### Q4 [10 points]. Evaluation
# 
# Calculate the accuracy (`tacc`), precision(`tprec`), recall(`trecall`), and Area Under the Receiver Operating Characteristic Curve (ROC AUC) `tauc` for the dataframe `df`.  The predicted class is in the column `pred` and the predicted probability from the model is in the column `pred_prob`. 

# In[ ]:


#Answer here



# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


#Please do not modify/delete this cell
assert tacc>0 and tacc<1.0
assert tprec>0 and tprec<1.0
assert trecall>0 and trecall<1.0


# In[ ]:


#Please do not modify/delete this cell


# ### Q5 Generate a Confusion Matrix
# 
# Use the the standard module confusion_matrix function from Scikit Learn to find the confusion matrix, setting it equal to `cm`. 
# 
# Parse the confusion matrix to set the true postives equal to `cm_tp` and the true negatives equal to `cm_tn`. 
# 
# Documentation 
# [https://scikit-learn.org/0.16/modules/generated/sklearn.metrics.confusion_matrix.html](https://scikit-learn.org/0.16/modules/generated/sklearn.metrics.confusion_matrix.html)
# 
# 
# 

# In[ ]:


from sklearn.metrics import confusion_matrix

# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


#Please do not modify/delete this cell
assert len(cm)==2
assert cm_tp>0
assert cm_tn>0


# ### q6 Evaluation - By Gender
# 
# Conduct the same analysis as above, but for the different subgroups of men and women.
# 
# ##### Calculate Metrics for Men
# Calculate accuracy (`tacc_m`) and Area Under the Receiver Operating Characteristic Curve (ROC AUC) `tauc_m` for only the men.
# 
# 
# ##### Calculate Metrics for Women
# Calculate accuracy (`tacc_w`) and Area Under the Receiver Operating Characteristic Curve (ROC AUC) `tauc_w` for only the women.

# In[ ]:


#Your solution here. 

# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


#Please do not modify/delete this cell
assert len(cm)==2
assert tacc_m>0
assert tacc_fm>0


# In[ ]:


#Please do not modify/delete this cell

