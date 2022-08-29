#!/usr/bin/env python
# coding: utf-8

# [![AnalyticsDojo](../fig/final-logo.png)](http://rpi.analyticsdojo.com)
# <center><h1>Intro to Tensorflow</h1></center>
# <center><h3><a href = 'http://rpi.analyticsdojo.com'>rpi.analyticsdojo.com</a></h3></center>
# 
# 

# 
# Adopted from [Hands-On Machine Learning with Scikit-Learn and TensorFlow by Aurélien Géron](https://github.com/ageron/handson-ml).
# 
# Apache License
# Version 2.0, January 2004
# http://www.apache.org/licenses/
# TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION
# 
# [For full license see repository.](https://github.com/ageron/handson-ml/blob/master/LICENSE)

# In[1]:


get_ipython().system('pip install tensorflow')


# In[16]:


# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


# ### Tensorflow Graph Creation
# - Tensorflow at it's base is trying to enable computation. 
# - Google designed tensorflow to scale, working with CPU, GPU, TPU 
# - Tensorflow separates the graph into a construction component and a computational component

# In[17]:



import tensorflow as tf
reset_graph()
x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x*x*y + y + 2
f


# ### Tensorflow Graph Computation
# - Two different syntax
# - Similar to other roles.

# In[18]:


# This is one way of creating a session. 
sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)
sess.close()
print(result)


# In[19]:


#This is like creating a file, we don't have to open and close. 
with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    result = f.eval()
print(result)


# In[20]:


#This is like creating a file, we don't have to open and close.
init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    result = f.eval()
print(result)


# # Managing Default Graphs
# - The default graph is the one that will be computed.

# In[21]:


reset_graph()

x1 = tf.Variable(1)
x1.graph is tf.get_default_graph()


# In[22]:


graph = tf.Graph()
with graph.as_default():
    x2 = tf.Variable(2)

x2.graph is graph


# In[23]:


x2.graph is tf.get_default_graph()


# #### The Graph Knows What Variables Are Related

# In[24]:


w = tf.constant(3)
x = w + 2
y = x + 5
z = x * 3

with tf.Session() as sess:
    print(y.eval())  # 10
    print(z.eval())  # 15


# In[25]:


with tf.Session() as sess:
    y_val, z_val = sess.run([y, z])
    print(y_val)  # 10
    print(z_val)  # 15


# # Linear Regression

# ## Regression Using the Normal Equation
# - The bias node in a neural network is a node that is always 'on'. That is, its value is set to 1 without regard for the data in a given pattern. It is analogous to the intercept in a regression model, and serves the same function.  
# 
# - You can read more about the normal equation [here]('https://en.wikipedia.org/wiki/Linear_least_squares_(mathematics)').

# In[26]:


import numpy as np
from sklearn.datasets import fetch_california_housing

#Reset the graph
reset_graph()

#Get the data
housing = fetch_california_housing()
m, n = housing.data.shape

#add bias term
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
housing_data_plus_bias


# In[27]:


#Do the math with Tensorflow. 
X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

with tf.Session() as sess:
    theta_tensorflow = theta.eval()

print(theta_tensorflow)


# Compare with pure NumPy

# In[28]:


X = housing_data_plus_bias
y = housing.target.reshape(-1, 1)
theta_numpy = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

print(theta_numpy)


# Compare with Scikit-Learn

# In[29]:


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing.data, housing.target.reshape(-1, 1))

print(np.r_[lin_reg.intercept_.reshape(-1, 1), lin_reg.coef_.T])


# In[ ]:





# In[ ]:





# In[ ]:




