#!/usr/bin/env python
# coding: utf-8

# [![AnalyticsDojo](https://github.com/rpi-techfundamentals/spring2019-materials/blob/master/fig/final-logo.png?raw=1)](http://rpi.analyticsdojo.com)
# <center><h1> Transfer Learning - NLP</h1></center>
# <center><h3><a href = 'http://rpi.analyticsdojo.com'>rpi.analyticsdojo.com</a></h3></center>

# This is adopted from: [Bag of Words Meets Bags of Popcorn](https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words)
# [https://github.com/wendykan/DeepLearningMovies](https://github.com/wendykan/DeepLearningMovies)
# 

# ## Transfer Learning - NLP
# 
# To be meaningfully modeled, words must be turned into Vectors.  This covers a number of the approaches for text vectorazation 1.0. 

# # Bag of Words

# In[1]:


import nltk
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
 
from gensim import similarities
import pandas as pd
import numpy as np
from gensim import models
# import custom filters
from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import strip_tags, strip_punctuation, strip_numeric, stem_text,  preprocess_string
from gensim.parsing.preprocessing import strip_multiple_whitespaces, strip_non_alphanum, remove_stopwords, strip_short
from gensim import corpora
from gensim.test.utils import common_corpus, common_dictionary
from gensim.similarities import MatrixSimilarity
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile
from gensim.models.doc2vec import TaggedDocument
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.dataset as ds
import pandas as pd
from pathlib import Path
from gensim.models import Phrases
from gensim.models.phrases import Phraser


# In[2]:


get_ipython().system('wget https://github.com/rpi-techfundamentals/spring2019-materials/raw/master/input/labeledTrainData.tsv')
get_ipython().system('wget https://github.com/rpi-techfundamentals/spring2019-materials/raw/master/input/unlabeledTrainData.tsv')
get_ipython().system('wget https://github.com/rpi-techfundamentals/spring2019-materials/raw/master/input/testData.tsv')


# In[3]:


train = pd.read_csv('labeledTrainData.tsv', header=0,                     delimiter="\t", quoting=3)
unlabeled_train= pd.read_csv('unlabeledTrainData.tsv', header=0,                     delimiter="\t", quoting=3)
test = pd.read_csv('testData.tsv', header=0,                     delimiter="\t", quoting=3)


# In[4]:


import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np


# In[5]:


print(train.columns.values, test.columns.values)


# In[6]:


train.head()


# In[7]:


print('The train shape is: ', train.shape)
print('The train shape is: ', test.shape)


# In[8]:


print('The first review is:')
print(train["review"][0])


# In[9]:


train


# In[10]:


import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

def embed_univ(df,column):
    encoder_lib_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    embed = hub.load(encoder_lib_url) # current encoder as at May 20th, 2021 - url "https://tfhub.dev/google/universal-sentence-encoder/4"
    message_embeddings = embed(df[column])
    df[column+'_universal'] = pd.Series(message_embeddings.numpy().tolist())
    return df
train2=embed_univ(train.iloc[0:10,:], 'review')
train2


# In[11]:


#title Configure the model { run: "auto" }
BERT_MODEL = "https://tfhub.dev/google/experts/bert/wiki_books/2" #
# Preprocessing must match the model, but all the above use the same.
PREPROCESS_MODEL = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

def embed_bert(df, column ):
    preprocess = hub.load(PREPROCESS_MODEL)
    bert = hub.load(BERT_MODEL)
    inputs = preprocess(df[column])
    outputs = bert(inputs)
    df[column+'_bert']=pd.Series(outputs["pooled_output"].numpy().tolist())
    return df
train2=embed_bert(train2.iloc[0:10,:], 'review')


# In[12]:


train2

