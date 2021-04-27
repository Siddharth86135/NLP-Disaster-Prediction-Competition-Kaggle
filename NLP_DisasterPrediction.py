# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:16:55 2020

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
    
    
#####################################################    
    
#  Different kinds of Vectorizers    
    
# Count Vectorizer
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1000)
X_train = cv.fit_transform(corpus).toarray()

# TF ID

from sklearn.feature_extraction.text import TfidfVectorizer
TV=TfidfVectorizer(ngram_range=(1,3),max_features=40000)
X_train= TV.fit_transform(corpus).toarray()  

# Hashing Vectorizer

hs_vectorizer=HashingVectorizer(non_negative=True,n_features=70000)
X_train=hs_vectorizer.fit_transform(corpus).toarray()


#####################################################

y_train = df_train['target']


#####################################################

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
  
X_test = TV.transform(corpus_test).toarray()  

#####################################################


from sklearn.model_selection import train_test_split
X_train_val, X_val, y_train_val, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

# Model

# Naive Bayes

from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()


x = 0
for alpha in np.arange(0,1,0.1):
    sub_classifier=MultinomialNB(alpha=alpha)
    sub_classifier.fit(X_train_val,y_train_val)
    y_pred_val=sub_classifier.predict(X_val)
    score = metrics.accuracy_score(y_val, y_pred_val)
    if (score > x):
        x = score
        print("Alpha: {}, Score : {}".format(alpha,x))

        
classifier = MultinomialNB(alpha = 0.4)
classifier.fit(X_train,y_train)        
y_pred=classifier.predict(X_test)

# Passive Aggresive Algorithm

from sklearn.linear_model import PassiveAggressiveClassifier
linear_clf = PassiveAggressiveClassifier(max_iter =1000, tol = 1e-3)

linear_clf.fit(X_train_val, y_train_val)
y_pred_val = linear_clf.predict(X_val)
score = metrics.accuracy_score(y_val, y_pred_val)
print("accuracy:   %0.3f" % score)

# Model Submission

pred=pd.DataFrame(y_pred)
Id_df=pd.read_csv('sample_submission.csv')

final_sub=pd.concat([Id_df['id'],pred],axis=1)
final_sub.columns=['id','target']

final_sub.to_csv('submission_LSTM.csv',index=False)















