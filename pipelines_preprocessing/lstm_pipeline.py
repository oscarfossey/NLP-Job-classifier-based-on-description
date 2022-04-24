# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 13:25:58 2022

@author: admin
"""

import os
os.system("git lfs install")
os.system("git clone https://huggingface.co/oscarfossey/job_classification")
os.system("pip install pickle")
os.system("pip install spacy")
os.system("spacy download fr_core_news_sm")
os.system("python -m spacy download fr_core_news_sm")
os.system("pip install joblib")
os.system("pip install keras")

import numpy as np
import nltk
import pickle
from keras.preprocessing.sequence import pad_sequences
import fr_core_news_sm
nltk.download('stopwords')
from joblib import load

global LSTM_tokenizer, stopwords, nlp, lstm_model
lstm_model = load(open("/content/job_classification/model_LSTM.joblib", 'rb')) 
stopwords = nltk.corpus.stopwords.words('french')
LSTM_tokenizer = pickle.load(open("/content/job_classification/LSTM_tokenizer", 'rb'))
nlp = fr_core_news_sm.load()

def preprocessing_LSTM(texts_array):
    """preprocessing the strings through the array to predict using the predict_tfidf function
    return an array of token inputs"""

    
    def preprocess(text):
      text = text.lower()
      text = text.replace('(', ' ').replace(')', ' ').replace('.', ' ').replace('  ', ' ')  #drop '(', ')', '.'
      text = nlp(text)
      words = [token.lemma_ for sent in text.sents for token in sent if not token.text in set(stopwords)]
      string = ' '.join(words)
      return string

    def tokenization_LSTM(new_offer):

      MAX_SEQUENCE_LENGTH=250
      seq = LSTM_tokenizer.texts_to_sequences([preprocess(new_offer)])
      padded = pad_sequences(seq, maxlen = MAX_SEQUENCE_LENGTH)
      
      return (padded)

    token_inputs = np.array([tokenization_LSTM(preprocess(txt)) for txt in list(texts_array.flatten())])

    return token_inputs


def predict_LSTM(texts_array):
  
  init_shape  = texts_array.shape

  predictions = np.array([lstm_model.predict(token_input)  for token_input in list(preprocessing_LSTM(texts_array.flatten()))])
  labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'N', 'M']
  predictions = np.array([labels[np.argmax(pred)] for pred in predictions])
  
  return predictions.reshape(init_shape)