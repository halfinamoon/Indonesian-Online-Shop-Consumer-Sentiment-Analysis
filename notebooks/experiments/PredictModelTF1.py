#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
# import bert
import subprocess as sp
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pandas as pd
import re
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, TFBertModel,  BertConfig, BertTokenizerFast
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix

# tensorflow.keras
from tensorflow.keras.layers import Input, Dropout, Dense, Embedding, LSTM, Flatten, Conv1D, MaxPooling1D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical


# In[2]:


# Label Name
predLabel= ['Negative', 'Positive']


# In[3]:


justifypred = lambda x: 1 if x>0.5 else 0


# In[4]:


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# In[5]:


def convertText2Ids(text):
    return tokenizer.encode_plus(
        text,
        max_length=512,
        add_special_tokens=True, # Add '[CLS]' and '[SEP]'
        return_token_type_ids=True,
        pad_to_max_length=True,
        return_attention_mask=True,
        truncation=True
    )


# In[6]:


print('Downloading BERT Tokenizer..')


# In[7]:


# BertTokenizer = None
# BertTokenizer = bert.bert_tokenization.FullTokenizer
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                            trainable=False)
vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = BertTokenizer(vocabulary_file, to_lower_case)


# In[8]:


# def textToToken(text):
#     #Train Data
#     return tokenizer(
#         text=text,
#         add_special_tokens=True,
#         max_length=200,
#         truncation=True,
#         padding=True, 
#         return_tensors='tf',
#         return_token_type_ids = False,
#         return_attention_mask = True,
#         verbose = True
#     )

def textToToken(text):
    #Train Data
    return tokenizer(
        text=text,
        add_special_tokens=True,
        max_length=200,
        truncation=True,
        padding='max_length', 
        return_tensors='tf',
        return_token_type_ids = False,
        return_attention_mask = True,
        verbose = True
    )


# In[14]:


# MODEL 1 FIX
TFmodel1 = tf.keras.models.load_model('Model/TFmodel1')
TFmodel1.load_weights('Model/TFmodel1/variables/variables')
# TFmodel1.load_weights()
query = str(input('Write some sentence: '))
query = textToToken(query)
# Add data dimension, model require input shape (1,200) 
# addDimen = np.expand_dims(query['input_ids'], axis=0)
#predict
singlePred = TFmodel1.predict(query['input_ids'])
print('Sentiment Prediction: ',predLabel[justifypred(singlePred)])

