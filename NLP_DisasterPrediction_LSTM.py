# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 19:16:49 2020

@author: sethi
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn import metrics
import numpy as np
import itertools

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')


df_train.drop(['id','keyword','location'],axis=1,inplace=True)
df_test.drop(['id','keyword','location'],axis=1,inplace=True)                           

print(df_train.info())
print(df_test.info())

import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# Data Cleaning
#As we know,twitter tweets always have to be cleaned before we go onto modelling.So we will do some basic
#cleaning such as spelling correction,removing punctuations,removing html tags and emojis etc.So let's start.

#Training Data

#Removing urls
def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)
df_train['text']=df_train['text'].apply(lambda x : remove_URL(x))

#Removing HTML tags
def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)
df_train['text']=df_train['text'].apply(lambda x : remove_html(x))

#Romoving Emojis
def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)
df_train['text']=df_train['text'].apply(lambda x: remove_emoji(x))

#Removing punctuations
import string
def remove_punct(text):
    table=str.maketrans('','',string.punctuation)
    return text.translate(table)
df_train['text']=df_train['text'].apply(lambda x : remove_punct(x))

#Spelling Correction
!pip install pyspellchecker
from spellchecker import SpellChecker
spell = SpellChecker()
def correct_spellings(text):
    corrected_text = []
    misspelled_words = spell.unknown(text.split())
    for word in text.split():
        if word in misspelled_words:
            corrected_text.append(spell.correction(word))
        else:
            corrected_text.append(word)
    return " ".join(corrected_text)
df_train['text']=df_train['text'].apply(lambda x : correct_spellings(x))

#Stemming and other preprocessing
corpus = []

for i in range(0, len(df_train)):
    review = re.sub('[^a-zA-Z]', ' ', df_train['text'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review) 
    
###############################################################
    
y_train = df_train['target']    
    
###############################################################

#Data Preparation for LSTM


import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense

voc_size=30000

# One Hot Representation
 
onehot_repr=[one_hot(words,voc_size)for words in corpus] 
a = 0
for i in onehot_repr:
    l = len(i)
    if (a < l):
        a = l
print(a) 



sent_length=23
embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)       

import numpy as np
X_final=np.array(embedded_docs)
y_final=np.array(y_train)

###############################################################

# Test Data

#Removing urls
def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)
df_test['text']=df_test['text'].apply(lambda x : remove_URL(x))

#Removing HTML tags
def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)
df_test['text']=df_test['text'].apply(lambda x : remove_html(x))

#Romoving Emojis
def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)
df_test['text']=df_test['text'].apply(lambda x: remove_emoji(x))

#Removing punctuations
import string
def remove_punct(text):
    table=str.maketrans('','',string.punctuation)
    return text.translate(table)
df_test['text']=df_test['text'].apply(lambda x : remove_punct(x))

#Spelling Correction
!pip install pyspellchecker
from spellchecker import SpellChecker
spell = SpellChecker()
def correct_spellings(text):
    corrected_text = []
    misspelled_words = spell.unknown(text.split())
    for word in text.split():
        if word in misspelled_words:
            corrected_text.append(spell.correction(word))
        else:
            corrected_text.append(word)
    return " ".join(corrected_text)
df_test['text']=df_test['text'].apply(lambda x : correct_spellings(x))

#Stemming and other preprocessing

corpus_test = []
for i in range(0, len(df_test)):
    review = re.sub('[^a-zA-Z]', ' ', df_test['text'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus_test.append(review)
    
    
##################################################################    


voc_size=30000


onehot_repr_test=[one_hot(words,voc_size)for words in corpus_test] 
b = 0
for i in onehot_repr_test:
    l = len(i)
    if (b < l):
        b = l
print(b) 



sent_length=23
embedded_docs_test =pad_sequences(onehot_repr_test,padding='pre',maxlen=sent_length)       

X_test = np.array(embedded_docs_test)


####################################################################   

from sklearn.model_selection import train_test_split
X_train_val, X_val, y_train_val, y_val = train_test_split(X_final, y_final, test_size=0.2, random_state=0)
    
####################################################################  

## Creating model

from tensorflow.keras.layers import Dropout
embedding_vector_features=50
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model.add(Dropout(0.3))
model.add(LSTM(100))
model.add(Dropout(0.3))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())


## Checking if tensorflow gpu is being used or not
print(tf.__version__)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

print(tf.test.is_built_with_cuda())

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")
    
    
from tensorflow.keras.layers import Dropout
embedding_vector_features=50
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model.add(Dropout(0.3))
model.add(LSTM(100))
model.add(Dropout(0.3))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())


### Finally Training
model.fit(X_train_val,y_train_val,validation_data=(X_val,y_val),
          epochs=40,batch_size=64)

## Predicting on Test Data
y_pred=model.predict_classes(X_test)    
    
    
#################################################

# Model Submission

pred=pd.DataFrame(y_pred)
Id_df=pd.read_csv('sample_submission.csv')

final_sub=pd.concat([Id_df['id'],pred],axis=1)
final_sub.columns=['id','target']

final_sub.to_csv('submission_LSTM.csv',index=False)    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    