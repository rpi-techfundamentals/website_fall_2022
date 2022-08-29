#!/usr/bin/env python
# coding: utf-8

# #### Homework-1
# ##### Total number of points: 50
# #### Due date: September 9th 2021
# 
# Before you submit this homework, make sure everything runs as expected. First, restart the kernel (in the menu, select Kernel → Restart) and then run all cells (in the menubar, select Cell → Run All). You can discuss with others regarding the homework but all work must be your own.
# 
# This homework will test your knowledge on basics of Python. The Python notebooks shared will be helpful to solve these problems.  
# 
# Steps to evaluate your solutions:
# 
# Step-1: Ensure you have installed Anaconda (Windows: https://docs.anaconda.com/anaconda/install/windows/ ; Mac:https://docs.anaconda.com/anaconda/install/mac-os/ ; Linux: https://docs.anaconda.com/anaconda/install/linux/)
# 
# Step-2: Open the Jupyter Notebook by first launching the anaconda software console
# 
# Step-3: Open the hw1.ipynb file and write your solutions at the appropriate location "# YOUR CODE HERE"
# 
# Step-4: You can restart the kernel and click run all (in the menubar, select Cell → Run All) on the center-right on the top of this window.
# 
# Step-5: Now go to "File" then click on "Download as" then click on "Notebook (.ipynb)" Please DO NOT change the file name and just keep it as "hw1.ipynb"
# 
# Step-6: Go to lms.rpi.edu and upload your homework at the appropriate link to submit this homework.
# 
# #### Please note that for any question in this assignment you will receive points ONLY if your solution passes all the test cases including hidden testcases as well. So please make sure you try to think all possible scenarios before submitting your answers.  
# - Note that hidden tests are present to ensure you are not hardcoding. 
# - If caught cheating: 
#     - you will receive a score of 0 for the 1st violation. 
#     - for repeated incidents, you will receive an automatic 'F' grade and will be reported to the dean of Lally School of Management. 

# #### Q1[5 points]. Assign the value for `x` to `150`. Set the value for `y` to `15` times `x` , and set the value for `z` to `y` divided by `x^2` (square of `x`).

# In[ ]:


#Assign x; y; z

# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


assert x==150
assert y==2250
assert z==0.1


# #### Q2-1 [5 points]: 
# 
# - Create an empty list `li1`
# - Create a list `li2` with integers 1, 2, 3
# - Create a nested list `li3` with `li1` and `li2`
# - Create a list `li4` with only `4` repeated 3 times.
# - Create a list `li5` with your firstName and lastName as two separate strings -- for example, li5 that I will build will be ['lydia', 'manikonda']
# 

# In[ ]:


#Need to create li1, li2, li3, li4, li5 according to the instructions

# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


assert len(li1) == 0
assert set(li2) == {1,2,3}
assert len(li3) == 2
assert len(set(li4)) == 1
assert len(li5)==2


# #### Q2-2 [5 points]: 
# 
# ###### Note that all are integers in this list. Please follow the exact order of execution here below from first line to the last line
# - Create a list `li1` with values in an order -- 10, -10, 20, -20 
# - Append a new element `30` to list `li1`  
# - Insert a new element `-30` at position `2`
# - Insert a new element `15` at position `20` (Yes it is 20 -- check what happens)

# In[ ]:


#Need to create li1 according to the instructions above

# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


assert len(li1)==7


# #### Q3-1 [5 points] 
# 
# ###### Create a dictionary `d1` with 5 keys 1, 2, 3, 4, 5 and their corresponding values as 2, 4, 6, 8, 10 respectively. 

# In[ ]:


#Creating the dictionary d1

# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


assert len(d1)==5
assert d1[3]==6
assert d1[5]==10


# #### Q3-2 [5 points] 
# 
# ###### Create a dictionary `d2` with 5 keys 1, 2, 3, 4, 5 and their corresponding values as 2, 4, 6, 8, 10 respectively. 
# ###### Add a new key-value pair to `d2` with key as `6` and value as `12`

# In[ ]:


#Creating the dictionary d2 and add a new key-value pair

# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


assert len(d2)==6


# #### Q3-3 [5 points].
# 
# - Create a list `l31` with numbers 1, 2, 3
# - Create another list `l32` with strings 'a', 'b', 'c'
# - Create a dictionary `d3` where length of `l31` is the key and `l32` is the value
# - Create a list `l33` that contains keys of dictionary `d3`

# In[ ]:


#Creating l31 and l32 and follow the instructions to create dictionary d3 

# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


assert len(l31)==3
assert set(l32)=={'a', 'b','c'}
assert len(d3)==1
assert len(l33)==1


# #### Q4-1 [5 points]. 
# 
# - Create a set `S1` with values 1, 2, 3 in the same order as stated. 
# - Assign the variable `Slen` as the length of this set `S1` 
# - Create another set `S2` with values 4, 5, 6 in the same order as stated. 
# - Create a new set `S3` by merging sets `S1` and `S2`

# In[ ]:


#Please follow the above instructions to create S1, S2, S3


# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


assert Slen==3
assert 1 in S1
assert S3=={1, 2, 3, 4, 5, 6}


# #### Q4-2 [5 points]
# 
# - Create a set `S4` with values 1, 2, 3, 4, 5 passed as strings 
# - Add another string `'6'` to `S4`
# - Convert each element in S4 which is a string to integer and save this new set as `S5`

# In[ ]:


# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


assert set(S4)=={'1', '2', '3', '4', '5','6'}
assert S5=={1, 2, 3, 4, 5, 6}


# #### Q5 [5 points]. Write a function that takes a number `num` (where `num` is >0) and returns a list of numbers from 0 to `num` (inclusive). If number `num`<=0 return `[-1]`. 

# In[ ]:



def PrintNums(num):

    # YOUR CODE HERE
    raise NotImplementedError()


# In[ ]:


assert PrintNums(3)==[0, 1, 2, 3]
assert PrintNums(4)==[0, 1, 2, 3, 4]
assert PrintNums(6)==[0, 1, 2, 3, 4, 5, 6]


# #### Q6 [5 points]. Write a function that takes two numbers `num1` and `num2`. Note that both `num1` and `num2` are `> 0`.  This function should return a list of numbers, where the length of the list is `num2`. 
# 
# ##### All the numbers in this list should be equal to `num1`. In other words, create a list that contains `num1` that is repeated `num2` number of times.
# ##### If `num1 <= 0` or `num2 <= 0` return `[-1]`

# In[ ]:


def PrintRepeatNums(num1, num2):
    # YOUR CODE HERE
    raise NotImplementedError()


# In[ ]:


assert PrintRepeatNums(3,2)==[3, 3]
assert PrintRepeatNums(2,4)==[2, 2, 2, 2]
assert PrintRepeatNums(6,1)==[6]

