#!/usr/bin/env python
# coding: utf-8

# [![AnalyticsDojo](https://github.com/rpi-techfundamentals/spring2019-materials/blob/master/fig/final-logo.png?raw=1)](http://rpi.analyticsdojo.com)
# <center><h1> Vectorization Options</h1></center>
# <center><h3><a href = 'http://rpi.analyticsdojo.com'>rpi.analyticsdojo.com</a></h3></center>

# This is adopted from: [Bag of Words Meets Bags of Popcorn](https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words)
# [https://github.com/wendykan/DeepLearningMovies](https://github.com/wendykan/DeepLearningMovies)
# 

# ## Vectorizors
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


# ### Common Preprocessing
# 
# Packages provide a variety of preprocessing routines. This results in a Tokenized set of data. 
# 
# 
# https://radimrehurek.com/gensim/parsing/preprocessing.html
# 

# In[ ]:


from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import strip_tags, strip_punctuation, strip_numeric, stem_text,  preprocess_string
from gensim.parsing.preprocessing import strip_multiple_whitespaces, strip_non_alphanum, remove_stopwords, strip_short

# define custom filters
text_col='review'
CUSTOM_FILTERS = [
                  lambda x: x.encode('utf-8').strip(),
                  lambda x: x.lower(), #lowercase
                  strip_multiple_whitespaces,# remove repeating whitespaces
                  strip_numeric, # remove numbers
                  strip_punctuation, #remove punctuation
                  remove_stopwords,# remove stopwordsß
                  stem_text # return porter-stemmed text,
                 ]

def preprocess(x, filters):
    results=preprocess_string(x, filters )
    return results

train[text_col+'_pro']=train[text_col].apply(preprocess, filters=CUSTOM_FILTERS)
train


# In[ ]:


from gensim import corpora
from gensim.test.utils import common_corpus, common_dictionary

def bow(x, dictionary):
    return dictionary.doc2bow(x)

#Create a Dictionary.
cdict = corpora.Dictionary(train[text_col+'_pro'].to_list())


#Create a Bag of Words Model
train[text_col+'_bow']=train[text_col+'_pro'].apply(bow, dictionary=  cdict)
train


# In[ ]:


def transform(x, model):
    return model[x]
    
tfidf_bow = models.TfidfModel( train[text_col+'_bow'].to_list(),  normalize=True)
train[text_col+'_tfidf_bow']=train[text_col+'_bow'].apply(transform, model=tfidf_bow )


# In[ ]:


#Word to Vec
train[text_col+'_tag']=pd.Series(TaggedDocument(doc, [i]) for i, doc in enumerate(train[text_col+'_pro'].to_list()))
doc2vec = Doc2Vec(train[text_col+'_tag'] , vector_size=50, window=2, min_count=1, workers=4)
train[text_col+'_docvecs']=pd.Series([doc2vec.docvecs[x] for x in range(len(train))])
train


# In[ ]:


def create_dense(x, vlen=50):
    try:
        x=dict(x)
        output=[]
        for i in range(vlen):
            if i in x.keys():
                output.append(np.float64(x[i]))
            else:
                output.append(0)
        return output
    except:
        return np.nan


# In[ ]:



lsi_model_bow = models.LsiModel(train[text_col+'_bow'].to_list(), id2word=cdict, num_topics=50)
train[text_col+'_lsi_bow']=train[text_col+'_bow'].apply(transform, model=lsi_model_bow)
train[text_col+'_lsi_bow_d']=train[text_col+'_lsi_bow'].apply(create_dense, vlen=50)


# In[ ]:


lsi_model_tfidf = models.LsiModel(train[text_col+'_tfidf_bow'].to_list(), id2word=cdict, num_topics=50)
train[text_col+'_lsi_tfidf']=train[text_col+'_tfidf_bow'].apply(transform, model=lsi_model_tfidf)
train[text_col+'_lsi_tfidf_d']=train[text_col+'_lsi_tfidf'].apply(create_dense, vlen=50)


# In[ ]:


lda_model_bow = models.LdaModel(train[text_col+'_bow'].to_list(), id2word=cdict, num_topics=50, minimum_probability=0)
train[text_col+'_lda_bow']=train[text_col+'_bow'].apply(transform, model=lda_model_bow)
train[text_col+'_lda_bow_d']=train[text_col+'_lda_bow'].apply(create_dense, vlen=50)


# In[ ]:



lda_model_tfidf = models.LdaModel(train[text_col+'_tfidf_bow'].to_list(), id2word=cdict, num_topics=50, minimum_probability=0)      
train[text_col+'_lda_tfidf']=train[text_col+'_tfidf_bow'].apply(transform, model=lda_model_tfidf)
train[text_col+'_lda_tfidf_d']=train[text_col+'_lda_tfidf'].apply(create_dense, vlen=50)


# In[ ]:


train

