#!/usr/bin/env python
# coding: utf-8

# [![AnalyticsDojo](https://github.com/rpi-techfundamentals/spring2019-materials/blob/master/fig/final-logo.png?raw=1)](http://rpi.analyticsdojo.com)
# <center><h1>Regression with Stats-Models </h1></center>
# <center><h3><a href = 'http://rpi.analyticsdojo.com'>rpi.analyticsdojo.com</a></h3></center>

# # Regression with Stats-Models

# In[18]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


# ## Scikit-learn vs Stats-Models
# - Scikit-learn provides framework which enables a similar api (way of interacting with codebase) for many different types of machine learning (i.e., predictive) models.
# - Stats-Models provices a clear set of results for statistical analsyses (understanding relationships) common to scientific (i.e., explanitory) models

# In[19]:


#Get a sample dataset
df = sm.datasets.get_rdataset("Guerry", "HistData").data


# ## About the Data
# - See [link](https://cran.r-project.org/web/packages/HistData/HistData.pdf).
# - Andre-Michel Guerry (1833) was the first to systematically collect and analyze social data on such things as crime, literacy and suicide with the view to determining social laws and the relations among these variables.

# In[20]:


df.columns


# In[21]:


df


# ## Predicting Gambling
# - `Lottery` Per capita wager on Royal Lottery. Ranked ratio of the proceeds bet on the royal lottery to population— Average for the years 1822-1826.  (Compte rendus par le ministre des finances)
# - `Literacy` Percent Read & Write: Percent of military conscripts who can read and write. 
# - `Pop1831` Population in 1831, taken from Angeville (1836), Essai sur la Statis-
# tique de la Population francais, in 1000s.

# In[22]:


#Notice this is an R style of Analsysis
results = smf.ols('Lottery ~ Literacy + np.log(Pop1831)', data=df).fit()


# In[23]:


print(results.summary())


# In[23]:





# ## Alternate Syntax
# This is a more pure way of 

# In[24]:


df['ln_Pop1831']=np.log(df['Pop1831'])


# In[25]:



X = df[['Literacy', 'ln_Pop1831']] # here we have 2 variables for the multiple linear regression. 
Y = df['Lottery']

X = sm.add_constant(X) # adding a constant
model = sm.OLS(Y, X).fit()
predictions = model.predict(X) 
print_model = model.summary()
print(print_model)
    


# ### Use Stargazer 
# 
# Show multiple different models easily using [Stargazer](https://github.com/mwburke/stargazer), a Python implementation of an R-package for implementing stepwise regression.
# 
# Let's add a different model. 

# In[26]:


get_ipython().system('pip install Stargazer')


# In[27]:


from stargazer.stargazer import Stargazer, LineLocation

X2 = df[['Literacy', 'ln_Pop1831','Crime_pers',	'Crime_prop']]
X2 = sm.add_constant(X2) # adding a constant
model2 = sm.OLS(Y, X2).fit()

stargazer = Stargazer([model,model2])
stargazer


# ## Challenge: Compare Results
# - Explore another model of `Lottery` and add it to the Stargazer results.  
# - Explore the stargazer documentation and customize the order of the variables, putting the constant and then the variables in all models on top (as is typically done). 
# 
# 
# 
