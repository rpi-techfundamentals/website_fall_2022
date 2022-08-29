#!/usr/bin/env python
# coding: utf-8

# #### Homework-3
# ##### Total number of points: 70
# #### Due date: September 27, 2021
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

# Use the titanic dataset from this url ('https://raw.githubusercontent.com/Spring2021IntrotoML/Spring2021IntrotoML.github.io/main/Lectures/titanic.csv') and we will use the same dataset in some of the questions here below. 

# #### Q1. Shape of a Data Frame. 
# 1. Import pandas package as pd.
# 2. Read the file from the 'url' and load the data into dataframe 'df' with default index.
# 3. Set number of rows equal to the variable `totalrows` and the number of columns equal to the variable `totalcols`. 
# 4. Print out the number of rows and columns, clearly labeling each.
# 
# 

# In[ ]:



url='https://raw.githubusercontent.com/Spring2021IntrotoML/Spring2021IntrotoML.github.io/main/Lectures/titanic.csv'

# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


assert totalrows==891
assert df['age'].sum()==21205.17


# In[ ]:


df


# Now we will be using the above dataframe df to do some preprocessing operations on this data. All the required libraries for further processing will be loaded here. 

# In[ ]:


import numpy as np
import pandas as pd
import re


# In[ ]:


df.columns
#df.groupby(['gender']).Class


# ### Q2 Dataframe Basic Analyses
# Determine how many first, second, and third class (assiging to the variables `class1`, `class2`, `class3`) passangers there are. 
# 
# *Hint -- Use value_counts operation* 
# 
# 

# In[ ]:


#Your answer here. 

# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


assert class1>0
assert class2>0
assert class3>0



# ### Q3 Groupby
# 
# Now use a groupby statement to calculate the mean age (use the 'age' attribute) of passengers who are are from each of the different passanger classes. 
# 
# Round the age to 2 decimal places (for example 3.14156 converts to 3.14 ) and assign the resulting variable to `class1_age`, `class1_age`, `class1_age`.
# 
# 

# In[ ]:


#Your answer 

# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


assert class1_age>0
assert isinstance(class1_age, np.floating)
assert class2_age>0 
assert isinstance(class1_age, np.floating)
assert class3_age>0
assert isinstance(class1_age, np.floating)


# ### Q4 Split Dataframe
# 
# Now split the dataframe `df` into two different dataframes `males` and `females`. 
# 
# 

# In[ ]:


#Your answer

# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


assert len(males['age'])==577
assert len(females['age'])==314


# #### Q5 Filter Missing Values
# 
# Now using 'females' remove all the rows where at least one element is missing and save this as a new dataframe object 'newfm'.

# In[ ]:



# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


assert newfm['age'].sum() == 2875.5
assert newfm['survived'].count() == 88


# ### Q6 Stratified sampling 
# 
# Utilize the original dataframe 'df' we created in Q1 to now perform 3 different subgroups of data based on the 'who' attribute.  There are 3 groups -- 'woman', 'man', 'child'. Perform random sampling to sample 5 data points (or rows) from each group and assign them to `woman_sam`, `man_sam` and `child_sam` -- these will look like dataframes as well.
# 
# *Hint: use the random package.

# In[ ]:


#Your answer

# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


assert set(women_sam['gender'])=={'female'}
assert set(man_sam['gender'])=={'male'}


# #### Q7  Feature Manipulation
# 1. Consider the original dataframe 'df'. Create a copy of this and call it `dfage`.
# 2. Remove all the rows of `dfage` in which the age value is missing. 
# 3. Then create a new column `dfage['age_st']` which replaces each value in the 'age' attribute with the corresponding standardized value *rounded to 2 decimal places*. You are not modifying `dfage['age']`.
# 
# 
# *Hint: See this for definition of standardized value.  [https://en.wikipedia.org/wiki/Standard_score](https://en.wikipedia.org/wiki/Standard_score) You can use np.mean() and np.std() functions to compute the mean and standard deviation values.*

# In[ ]:




# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


assert dfage.shape==(714, 16)
assert dfage['age'].isna().sum()==0


# #### Q8 Feature Creation 2
# 1. Re-read the file and create 'df'.
# 2. Remove all the rows with at least one missing value. 
# 3. Create a column `stown` in `df` using `embark_town`.  For the `embark_town` is 'Southampton' make `stown` 1. Otherwise `stown` is 0.

# In[ ]:



url='https://raw.githubusercontent.com/Spring2021IntrotoML/Spring2021IntrotoML.github.io/main/Lectures/titanic.csv'

# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:



assert set(df['stown'])=={0,1}
assert df.shape==(182, 16)


# ### Q9. Beautiful Soup
# Use the html content shared here below and parse it using the 'soup' object to 
# assign all the unique 'Hometown' values to as a list object `hts`
# 
# 
# Strip any leading or trailing white space and convert each value of the hometown in `hts` to a lower-case.
# 
# 
# Print the final answer.

# In[ ]:




import requests
from bs4 import BeautifulSoup
import operator
import pandas as pd
import json
newtext = """
<p>
    <strong class="person1">YOB:</strong> 1990<br />
    <strong class="person1">GENDER:</strong> FEMALE<br />
    <strong class="person1">EYE COLOR:</strong> GREEN<br />
    <strong class="person1">HAIR COLOR:</strong> BROWN<br />
    <strong class="person1">GPA:</strong> 4<br />
    <strong class="person1">Hometown:</strong> Minneapolis<br />
</p>

<p>
    <strong class="person2">YOB:</strong> 1993<br />
    <strong class="person2">GENDER:</strong> FEMALE<br />
    <strong class="person2">EYE COLOR:</strong> BROWN<br />
    <strong class="person2">HAIR COLOR:</strong> BLACK<br />
    <strong class="person2">GPA:</strong> 3.5<br />
    <strong class="person2">Hometown:</strong> New York<br />
</p>

"""
hts=[]
soup = BeautifulSoup(newtext)
# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


assert len(hts)==2
assert hts==['minneapolis', 'new york']


# ### Q10 String operations
# 
# 1. Given a string `str10` first split the string into words, strip any leading or trailing white space, and convert them to lowercase. 
# 2. Now using the 'join' operation we learnt in the class, concatenate these words to a new string `str11` with a '-' between each.
# 
# For example: str1 is 'it is cold today' to str2 will be: 'it-is-cold-today'
# 

# In[ ]:



str10 = 'Email the company at xyz@abcd.com and is the easiest way compared to tweet @abcdxyz'
# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


assert len(str11)==83


# ### Q11. Regular Expressions
# 
# Use the regular expressions package to extract all the email ids mentioned in the above sentence 'str10' as a list `emails`.
# 
# 

# In[ ]:


###Q11. Use the regular expressions package to extract all the email ids mentioned in the above sentence 'str10'
### as a list 'emails'
import re
str10 = 'Email the company at xyz@abcd.co and is the easiest way compared to tweet @abcdxyz'
emails=[]
# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


assert len(emails)==1
assert emails[0]=='xyz@abcd.co'

