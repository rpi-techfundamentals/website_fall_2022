#!/usr/bin/env python
# coding: utf-8

# 
# [![AnalyticsDojo](https://github.com/rpi-techfundamentals/spring2019-materials/blob/master/fig/final-logo.png?raw=1)](http://rpi.analyticsdojo.com)
# <center><h1>Titanic Classification</h1></center>
# <center><h3><a href = 'http://introml.analyticsdojo.com'>introml.analyticsdojo.com</a></h3></center>
# 
# 

# # Titanic Classification

# As an example of how to work with both categorical and numerical data, we will perform survival predicition for the passengers of the HMS Titanic.
# 

# In[56]:


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

# In[57]:


train.head()


# ### Preprocessing function
# 
# We want to create a preprocessing function that can address transformation of our train and test set.  

# In[58]:


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

# In[59]:


#Import Module
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=122,stratify=y)


# In[60]:


print(train_y.mean(), val_y.mean())


# In[61]:


from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import metrics


# In[62]:


from sklearn import tree
classifier = tree.DecisionTreeClassifier(max_depth=4)
#This fits the model object to the data.
classifier.fit(train_X, train_y)
#This creates the prediction. 
train_y_pred = classifier.predict(train_X)
val_y_pred = classifier.predict(val_X)
test['Survived'] = classifier.predict(test_X)
print("Metrics score train: ", metrics.accuracy_score(train_y, train_y_pred) )
print("Metrics score validation: ", metrics.accuracy_score(val_y, val_y_pred) )


# In[63]:


print("Metrics score train: ", metrics.recall_score(train_y, train_y_pred) )
print("Metrics score validation: ", metrics.recall_score(val_y, val_y_pred) )


# ### Outputting Probabilities
# Some evaluation metrics (like the [Area Under the Receiver Operating Characteristic Curve (ROC AUC)](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html) take the probability rather than the class which is output by the model.  
# 
# 
# The function `predict_proba` outputs the probability of each class. Here, we want only the second value which is the probability of survived.
# 
# 
# **When working with a new evaluation metric, always check to see whether it takes the probability or the class.**

# In[64]:


train_y_pred_prob = classifier.predict_proba(train_X)[:,1]
val_y_pred_prob = classifier.predict_proba(val_X)[:,1]
test_y_pred_prob = classifier.predict_proba(test_X)[:,1]


# In[65]:


print("Metrics score train: ", metrics.roc_auc_score(train_y, train_y_pred_prob) )
print("Metrics score validation: ", metrics.roc_auc_score(val_y, val_y_pred_prob) )


# In[66]:


test[['PassengerId','Survived']].to_csv('submission.csv')
from google.colab import files
files.download('submission.csv')


# ## Challenge
# Create a function that can accept any Scikit learn model and assess the perfomance in the validation set, storing results as a dataframe. 

# In[67]:



#Function Definition

def evaluate(name, dtype, y_true, y_pred, y_prob, results=pd.Series(dtype=float)):
  """
  This creates a Pandas series with different results. 
  """
  results['name']=name
  results['accuracy-'+dtype]=metrics.accuracy_score(y_true, y_pred)
  results['recall-'+dtype]=metrics.recall_score(y_true, y_pred)
  results['auc-'+dtype]=metrics.roc_auc_score(y_true, y_prob)
  return results


def model(name, classifier, train_X, train_y, val_X, val_y):
  """
  This will train and evaluate a classifier. 
  """
  classifier.fit(train_X, train_y)
  #This creates the prediction. 
  r1= evaluate(name, "train", train_y, classifier.predict(train_X), classifier.predict_proba(train_X)[:,1])
  r1= evaluate(name,"validation", val_y, classifier.predict(val_X), classifier.predict_proba(val_X)[:,1], results=r1)
  return r1



# ## Analyze Multiple Models
# 
# This code will model all values which are in the dictionary. 
# 
# 

# In[68]:


final=pd.DataFrame()
allmodels={"knearest": KNeighborsClassifier(n_neighbors=10),
           "adaboost":AdaBoostClassifier()}

for key, value in  allmodels.items():
  print("Modeling: ", key, "...")
  results= model(key, value, train_X, train_y, val_X, val_y)
  final=final.append(results, ignore_index=True)
final_order=['name','accuracy-train', 'accuracy-validation', 'auc-train', 'auc-validation','recall-train', 'recall-validation']
final=final.loc[:,final_order]
final


# ### Challenge 
# 
# Augment the modeling to include [Random Forests](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) at multiple different hyperparameter levels. 
# 
# 
# Augment the evaluation to include [Balanced Accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html) and [F1](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) score.
# 
