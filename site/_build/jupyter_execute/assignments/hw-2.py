#!/usr/bin/env python
# coding: utf-8

# #### Homework-2
# ##### Total number of points: 70
# #### Due date: September 16th 2021
# 
# Before you submit this homework, make sure everything runs as expected. First, restart the kernel (in the menu, select Kernel → Restart) and then run all cells (in the menubar, select Cell → Run All). You can discuss with others regarding the homework but all work must be your own.
# 
# This homework will test your knowledge on the basics of Python including functions and packages such as numpy and pandas. In-class exercises and different Python notebooks shared on Piazza will be helpful to solve these problems.  
# 
# Steps to evaluate your solutions:
# 
# Step-1: Ensure you have installed Anaconda (Windows: https://docs.anaconda.com/anaconda/install/windows/ ; Mac:https://docs.anaconda.com/anaconda/install/mac-os/ ; Linux: https://docs.anaconda.com/anaconda/install/linux/)
# 
# Step-2: Open the Jupyter Notebook by first launching the anaconda software console
# 
# Step-3: Open the homework's .ipynb file and write your solutions at the appropriate location "# YOUR CODE HERE"
# 
# Step-4: You can restart the kernel and click run all (in the menubar, select Cell → Run All) on the center-right on the top of this window.
# 
# Step-5: Now go to "File" then click on "Download as" then click on "Notebook (.ipynb)" Please DO NOT change the file name and just keep it as "Homework-2.ipynb"
# 
# Step-6: Go to lms.rpi.edu and upload your homework at the appropriate link to submit this homework.
# 
# #### Please note that for any question in this assignment you will receive points ONLY if your solution passes all the test cases including hidden testcases as well. So please make sure you try to think all possible scenarios before submitting your answers.  
# - Note that hidden tests are present to ensure you are not hardcoding. 
# - If caught cheating: 
#     - you will receive a score of 0 for the 1st violation. 
#     - for repeated incidents, you will receive an automatic 'F' grade and will be reported to the dean of Lally School of Management. 

# #### Q1 [10 points]. Write a function that takes a number and checks if it is a prime number. Return 1 if its a prime number or else 0. 
# - A number is prime if there are no other factors except 1 and itself. 
# - The program returns -- 1, if it is a prime number; otherwise 0.

# In[ ]:



def isPrimeNum(num):
    '''Write a program to check if a given number is a prime number or not'''
    # YOUR CODE HERE
    
    


# In[ ]:


isPrimeNum(1)==0
isPrimeNum(10)==0
isPrimeNum(97)==1


# #### Q2 [5 points]. import the numpy package as np. 
# 

# In[ ]:



'''Note that ONLY if you get this solution right all the below questions will be executed'''

# YOUR CODE HERE


# In[ ]:


assert np.prod([1,2,3])==6


# #### Q3 [5 points] Create a numpy array called `arr3` of all zeros except the fifth value is 1 --  and is of length 10. 

# In[ ]:


'''Create arr3 according to the instructions above'''

# YOUR CODE HERE




# In[ ]:


assert isinstance(arr3, np.ndarray)
assert arr3[:4].all() == 0
assert arr3[4] == 1


# #### Q4 [5 points] Create a numpy array 'arr4' with elements 1,3,5,7,9. Then create a new array 'arr41' whose elements are squared values of each element in 'arr4'.
# 

# In[ ]:


#Now create a new array 'arr41' whose elements are squared values of each element in 'arr4'

# YOUR CODE HERE



# In[ ]:


assert isinstance(arr4, np.ndarray)
assert arr4[2] == 5
assert arr41[2] == 25
assert arr41[1] == 9


# #### Q5 [10 points]. Return the element that occurs maximum number of times in a numpy array
# 

# In[ ]:



def maxElem(arr5):
    """Returns the element that occurs maximum number of times
    """
    # YOUR CODE HERE
    


# In[ ]:


'''case-1'''
list3=np.array([1])
assert maxElem(list3)==1

'''case-2'''
list3=np.array([1,2,3,1,2,3,1,1])
assert maxElem(list3)==1


# In[ ]:


import pandas as pd


# #### Q6 [5 points]. Please follow the instructions very carefully. 
# #### Create a list 'l6' (lowercase 'L')and assign 1, 2, 3, 4, 5 to this list
# #### Use l6 to create a series S6 with a default index
# 

# In[ ]:


#Creating l6

# YOUR CODE HERE


# In[ ]:


assert len(l6)==5
assert max(l6)==5
assert S6[3]==4


# #### Q7 [10 points]. This function takes a list 'l7' as an input and creates 2 more lists ('l71' and 'l72') by manipulating 'l7' as follows
# -  l71 -- copy l7 into l71 and add the 1st element of l7 to the end of l71
# - l72 -- copy l71 into l72 and add the mean value of l71 to the end of l72
# - Now use 'l7', 'l71' and 'l72' to create a data frame with corresponding column names as 'first', 'second' and 'third' 
# - return this dataframe

# In[ ]:



def createDF(l7):
    """Returns the dataframe that is built using an original list
    """
    # YOUR CODE HERE
    


# In[ ]:


'''case-1'''
list7=[2,4,6,8]
assert list(createDF(list7)['second'])[:5]==[2.0,4.0,6.0,8.0,2.0]

'''case-2'''
list7=[2,4,6,8]
assert list(createDF(list7)['first'])[:4]==[2,4,6,8]


# #### Q8 [10 points]. Read the .csv file from this url and load it into dataframe 'df'
# #### Using pandas dataframe operations, output the total number of rows with `survived`==1 and `who` is `woman` as `var81`.
# #### Using pandas dataframe operations, output the total number of rows with `survived`==1 and `who` is `child` as  `var82`. respectively.
# 

# In[ ]:


url='https://raw.githubusercontent.com/lmanikon/lmanikon.github.io/master/teaching/datasets/titanic.csv'

# YOUR CODE HERE


# In[ ]:


assert df['age'].sum()==21205.17
assert df.isnull().sum().sum()==869
assert var82==49


# #### Q9 [10 points]. Read the .csv file from this url and load it into dataframe 'df'
# #### Using groupby operation over `embark_town` and `gender`, assign `var9` as the total number of people survived (`survived` is 1) where they embarked (`embark_town`) at `Southampton` and are female (`gender` is `female`). 
# 

# In[ ]:


url='https://raw.githubusercontent.com/lmanikon/lmanikon.github.io/master/teaching/datasets/titanic.csv'

import pandas as pd
# YOUR CODE HERE


# In[ ]:


assert df['age'].sum()==21205.17
assert df.isnull().sum().sum()==869
assert var9==140

