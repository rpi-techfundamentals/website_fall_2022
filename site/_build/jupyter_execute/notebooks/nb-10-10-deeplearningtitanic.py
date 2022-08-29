#!/usr/bin/env python
# coding: utf-8

# 
# [![AnalyticsDojo](https://github.com/rpi-techfundamentals/spring2019-materials/blob/master/fig/final-logo.png?raw=1)](http://rpi.analyticsdojo.com)
# <center><h1>Titanic Classification - Keras API</h1></center>
# <center><h3><a href = 'http://introml.analyticsdojo.com'>introml.analyticsdojo.com</a></h3></center>
# 
# 

# # Titanic Classification - Deep Learning Tensorflow

# As an example of how to work with both categorical and numerical data, we will perform survival predicition for the passengers of the HMS Titanic.
# 

# In[ ]:


import os
import pandas as pd
train = pd.read_csv('https://raw.githubusercontent.com/rpi-techfundamentals/spring2019-materials/master/input/train.csv')
test = pd.read_csv('https://raw.githubusercontent.com/rpi-techfundamentals/spring2019-materials/master/input/test.csv')

print(train.columns, test.columns)


# Here is a broad description of the keys and what they mean:
# 
# ```
# pclass          Passenger Class
#                 (1 = 1st; 2 = 2nd; 3 = 3rd)
# survival        Survival
#                 (0 = No; 1 = Yes)
# name            Name
# sex             Sex
# age             Age
# sibsp           Number of Siblings/Spouses Aboard
# parch           Number of Parents/Children Aboard
# ticket          Ticket Number
# fare            Passenger Fare
# cabin           Cabin
# embarked        Port of Embarkation
#                 (C = Cherbourg; Q = Queenstown; S = Southampton)
# boat            Lifeboat
# body            Body Identification Number
# home.dest       Home/Destination
# ```
# 
# In general, it looks like `name`, `sex`, `cabin`, `embarked`, `boat`, `body`, and `homedest` may be candidates for categorical features, while the rest appear to be numerical features. We can also look at the first couple of rows in the dataset to get a better understanding:

# In[ ]:


train.head()


# ### Preprocessing function
# 
# We want to create a preprocessing function that can address transformation of our train and test set.  

# In[ ]:


from sklearn.impute import SimpleImputer
import numpy as np

cat_features = ['Pclass', 'Sex', 'Embarked']
num_features =  [ 'Age', 'SibSp', 'Parch', 'Fare'  ]


def preprocess(df, num_features, cat_features, dv):
    features = cat_features + num_features
    if dv in df.columns:
      y = df[dv]
    else:
      y=None 
    #Address missing variables
    print("Total missing values before processing:", df[features].isna().sum().sum() )
  
    imp_mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    df[cat_features]=imp_mode.fit_transform(df[cat_features] )
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    df[num_features]=imp_mean.fit_transform(df[num_features])
    print("Total missing values after processing:", df[features].isna().sum().sum() )
   
    X = pd.get_dummies(df[features], columns=cat_features, drop_first=True)
    return y,X

y, X =  preprocess(train, num_features, cat_features, 'Survived')
test_y, test_X = preprocess(test, num_features, cat_features, 'Survived')


# ### Train Test Split
# 
# Now we are ready to model. We are going to separate our Kaggle given data into a "Train" and a "Validation" set. 
# 
# 

# In[ ]:


#Import Module
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=122,stratify=y)


# In[ ]:


train_X.shape


# ### Sequential Model Classification. 
# 
# This is our training. We do all of the preprocessing our old way and just use the dataframe.values to pass to Keras.
# 
# https://keras.io/guides/sequential_model/
# 

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras import metrics


#Create our model using sequential mode
model = Sequential()
model.add(Dense(20, input_dim=9, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()





# In[ ]:


#Specify the model 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Fit the model
model.fit(train_X.values, train_y.values, epochs=100, batch_size=20, verbose=2)

_, trainperf = model.evaluate(train_X, train_y)
_, testperf = model.evaluate(val_X, val_y)


# In[ ]:


# Alternate Sequential syntax
import tensorflow as tf
altmodel = tf.keras.Sequential([
    tf.keras.layers.Dense(20, input_dim=9, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(1)
])
altmodel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
altmodel.summary()



# In[ ]:



#Specify the model 
#Fit the model
altmodel.fit(train_X.values, train_y.values, epochs=100, batch_size=20, verbose=2)

_, altmodelTrainperf = altmodel.evaluate(train_X, train_y)
_, altmodelValPerf = altmodel.evaluate(val_X, val_y)


# ## Functional Model 
# 
# https://keras.io/guides/functional_api/

# In[ ]:


inputs = tf.keras.Input(shape=(9,))
x = tf.keras.layers.Dense(20, activation=tf.nn.relu)(inputs)
x = tf.keras.layers.Dense(100, activation=tf.nn.relu)(x)
outputs = tf.keras.layers.Dense(1)(x)
modelalt2 = tf.keras.Model(inputs=inputs, outputs=outputs, name="classifier")
modelalt2.summary()


# # The Keras Model Subclassing Methods.
# 
# https://keras.io/api/models/model/

# In[ ]:


import tensorflow as tf

class MyModel(tf.keras.Model):

  def __init__(self):
    super(MyModel, self).__init__()
    self.dense1 = tf.keras.layers.Dense(20, input_dim=9, activation=tf.nn.relu)
    self.dense2 = tf.keras.layers.Dense(100, activation=tf.nn.relu)
    self.dense3 = tf.keras.layers.Dense(1)

  def call(self, inputs):
    x = self.dense1(inputs)
    x = self.dense2(x)
    return self.dense3(x)

altmodel3 = MyModel()


# In[ ]:


altmodel3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#Fit the model
altmodel3.fit(train_X.values, train_y.values, epochs=100, batch_size=20, verbose=2)

altmodel3.summary()


# 
