[![AnalyticsDojo](https://github.com/rpi-techfundamentals/spring2019-materials/blob/master/fig/final-logo.png?raw=1)](http://introml.analyticsdojo.com)
<center><h1>Introduction to Python - Null Values</h1></center>
<center><h3><a href = 'http://introml.analyticsdojo.com'>introml.analyticsdojo.com</a></h3></center>




# Null Values

## Running Code using Kaggle Notebooks
- Kaggle utilizes Docker to create a fully functional environment for hosting competitions in data science.
- You could download/run this locally or run it online. 
- Kaggle has created an incredible resource for learning analytics.  You can view a number of *toy* examples that can be used to understand data science and also compete in real problems faced by top companies. 

!wget https://raw.githubusercontent.com/rpi-techfundamentals/spring2019-materials/master/input/train.csv
!wget https://raw.githubusercontent.com/rpi-techfundamentals/spring2019-materials/master/input/test.csv

### Null Values Typical When Working with Real Data
- Null values `NaN` in Pandas


import numpy as np 
import pandas as pd 

# Input data files are available in the "../input/" directory.
# Let's input them into a Pandas DataFrame
train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")

print(train.dtypes)

train.head()

test.head()

#Let's get some general s
totalRows=len(train.index)
print("There are ", totalRows, " so totalRows-count is equal to missing variables.")
print(train.describe())
print(train.columns)


# We are going to do operations on thes to show the number of missing variables. 
train.isnull().sum()

### Dropping NA
- If we drop all NA values, this can dramatically reduce our dataset.  
- Here while there are 891 rows total, there are only 183 complete rows
- `dropna()` and `fillna()` are 2 method for dealing with this, but they should be used with caution.
- [Fillna documentation](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.fillna.html)
- [Dropna documentation](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.dropna.html)

# This will drop all rows in which there is any missing values
traindrop=train.dropna()
print(len(traindrop.index))
print(traindrop.isnull().sum())


# This will drop all rows in which there is any missing values
trainfill=train.fillna(0)  #This will just fill all values with nulls.  Probably not what we want. 
print(len(trainfill.index))
print(traindrop.isnull().sum())

# forward-fill previous value forward.
train.fillna(method='ffill')

# forward-fill previous value forward.
train.fillna(method='bfill')

### Customized Approach
- While those approaches

average=train.Age.mean()
print(average)

#Let's convert it to an int
average= int(average)
average

#This will select out values that  
train.Age.isnull()

#Now we are selecting out those values 
train.loc[train.Age.isnull(),"Age"]=average
train

### More Complex Models - Data Imputation
- Could be that Age could be inferred from other variables, such as SibSp, Name, Fare, etc.  
- A next step could be to build a more complex regression or tree model that would involve data tat was not null.

### Missing Data - Class Values
- We have 2 missing data values for the Embarked Class
- What should we replace them as?


pd.value_counts(train.Embarked)

train.Embarked.isnull().sum()

train[train.Embarked.isnull()]

train.loc[train.Embarked.isnull(),"Embarked"]="S"



This work is licensed under the [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/) license agreement.
Adopted from [materials](https://github.com/phelps-sg/python-bigdata) Copyright [Steve Phelps](http://sphelps.net) 2014