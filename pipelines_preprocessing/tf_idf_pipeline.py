import os
os.system("git lfs install")
os.system("git clone https://huggingface.co/oscarfossey/job_classification")
os.system("pip install pickle")
os.system("pip install spacy")
os.system("spacy download fr_core_news_sm")
os.system("python -m spacy download fr_core_news_sm")


import numpy as np
import nltk
import pickle
import fr_core_news_sm

nltk.download('stopwords')

global stopwords, nlp, tf_idf_over, naive_bayes_classifier
stopwords = nltk.corpus.stopwords.words('french')
nlp = fr_core_news_sm.load()
tf_idf_over = pickle.load(open("/content/job_classification/tf_idf_over", 'rb'))
naive_bayes_classifier_over = pickle.load(open("/content/job_classification/naive_bayes_classifier_over", 'rb'))


def preprocessing_tfidf(texts_array):
    """preprocessing the strings through the array to predict using the predict_tfidf function
    return an array of string"""

    init_shape  = texts_array.shape
    
    def preprocess(text):
      text = text.lower()
      text = text.replace('(', ' ').replace(')', ' ').replace('.', ' ').replace('  ', ' ')  #drop '(', ')', '.'
      text = nlp(text)
      words = [token.lemma_ for sent in text.sents for token in sent if not token.text in set(stopwords)]
      string = ' '.join(words)
      return string

    preprocessed_text = np.array([preprocess(txt) for txt in list(texts_array.flatten())])

    return preprocessed_text.reshape(init_shape)

def predict_tfidf(texts_array):
    """Draw prediction for each text of the array after preprocessing them
    return an array of the prediction the same size as input array"""
    
    predictions = naive_bayes_classifier_over.predict(tf_idf_over.transform(preprocessing_tfidf(texts_array).flatten()))

    return predictions.reshape(texts_array.shape)