#!/usr/bin/env python
# coding: utf-8

# [![AnalyticsDojo](https://github.com/rpi-techfundamentals/spring2019-materials/blob/master/fig/final-logo.png?raw=1)](http://rpi.analyticsdojo.com)
# <center><h1>Regression with Tensorflow/Keras </h1></center>
# <center><h3><a href = 'http://rpi.analyticsdojo.com'>rpi.analyticsdojo.com</a></h3></center>

# # Regression with Tensorflow/Keras

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr


get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina' #set 'png' here when working on notebook")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


get_ipython().system('wget https://raw.githubusercontent.com/rpi-techfundamentals/spring2019-materials/master/input/boston_test.csv && wget https://raw.githubusercontent.com/rpi-techfundamentals/spring2019-materials/master/input/boston_train.csv')


# In[ ]:


train = pd.read_csv("boston_train.csv")
test = pd.read_csv("boston_test.csv")


# In[ ]:


train.head()


# In[ ]:


all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))


# ### Data preprocessing: 
# We're not going to do anything fancy here: 
#  
# - First I'll transform the skewed numeric features by taking log(feature + 1) - this will make the features more normal    
# - Create Dummy variables for the categorical features    
# - Replace the numeric missing values (NaN's) with the mean of their respective columns

# In[ ]:


matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
prices = pd.DataFrame({"price":train["SalePrice"], "log(price + 1)":np.log1p(train["SalePrice"])})
prices.hist()


# In[ ]:


#log transform the target:
train["SalePrice"] = np.log1p(train["SalePrice"])

#log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])


# In[ ]:


all_data = pd.get_dummies(all_data)


# In[ ]:


#filling NA's with the mean of the column:
all_data = all_data.fillna(all_data.mean())


# In[ ]:


#creating matrices for sklearn:
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y_train = train.SalePrice


# In[ ]:





# ### Models Deep Learning Models
# 
# Now we are going to use Deep Learning models

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras import metrics
import tensorflow as tf

#Create our model using sequential mode
model = Sequential()
model.add(tf.keras.layers.Normalization(axis=-1))
model.add(Dense(units=1))
model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')

model.fit( X_train, y_train, epochs=100, verbose=2, validation_split = 0.2)
model.summary()


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras import metrics
import tensorflow as tf

def r2(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

#Create our model using sequential mode
model = Sequential()
model.add(tf.keras.layers.Normalization(axis=-1))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_absolute_error', metrics=[r2])

model.fit( X_train, y_train, epochs=100, verbose=2, validation_split = 0.2)
model.summary()


# In[ ]:


# Alternate Sequential syntax, with some additional data
import tensorflow as tf
altmodel = tf.keras.Sequential([
    tf.keras.layers.Normalization(axis=-1),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(units=1),
])
altmodel.compile(loss='mean_absolute_error', optimizer='adam', metrics=[r2])
altmodel.fit( X_train, y_train, epochs=100, verbose=2, validation_split = 0.2)
altmodel.summary()
#Can predict and then evaluate. 
y_pred_train_alt = altmodel.predict(X_train)
y_pred_test_alt = altmodel.predict(X_test)


# In[ ]:


from sklearn import metrics as skmetrics

deep_r2_train=skmetrics.r2_score(y_train, y_pred_train_alt)
deep_r2_train


# 
