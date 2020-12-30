
abc = 'Mert'

import os

import sys
import pandas as pd
import numpy as np


import pickle

def save_model(model, filename):
  pickle_filename = "./code/saved_models/" + filename + ".pkl"
  os.makedirs(os.path.dirname(pickle_filename), exist_ok=True)
  picklefile = open(pickle_filename, 'wb')
  pickle.dump(model, picklefile)
  picklefile.close()

def load_model(model_name):
  pickle_filename = "./code/saved_models/" + model_name + ".pkl"
  picklefile = open(pickle_filename, 'rb')
  model = pickle.load(picklefile)
  picklefile.close()
  return model




import re

import nltk


from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

from text_preprocessing import preprocess_text, remove_stopword
from text_preprocessing import to_lower, remove_email, remove_url, lemmatize_word, remove_punctuation, check_spelling, expand_contraction, remove_name, remove_number, remove_special_character, remove_punctuation, remove_whitespace, normalize_unicode, remove_stopword, preprocess_text

# A list of contractions from http://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
contractions = { 
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he's": "he is",
"how'd": "how did",
"how'll": "how will",
"how's": "how is",
"i'd": "i would",
"i'll": "i will",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'll": "it will",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"must've": "must have",
"mustn't": "must not",
"needn't": "need not",
"oughtn't": "ought not",
"shan't": "shall not",
"sha'n't": "shall not",
"she'd": "she would",
"she'll": "she will",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"that'd": "that would",
"that's": "that is",
"there'd": "there had",
"there's": "there is",
"they'd": "they would",
"they'll": "they will",
"they're": "they are",
"they've": "they have",
"wasn't": "was not",
"we'd": "we would",
"we'll": "we will",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"where'd": "where did",
"where's": "where is",
"who'll": "who will",
"who's": "who is",
"won't": "will not",
"wouldn't": "would not",
"you'd": "you would",
"you'll": "you will",
"you're": "you are"
}

def clean_text(text, remove_stopwords=True, remove_whitespaces=False):
  preprocess_functions = [to_lower, remove_email, remove_url, remove_punctuation, lemmatize_word, check_spelling, expand_contraction, remove_name, remove_stopword]
  # text = preprocess_text(text, preprocess_functions)

  # Convert words to lower case
  # text = text.lower()
  
  # Replace contractions with their longer forms 
  if True:
      text = text.split()
      new_text = []
      for word in text:
          if word in contractions:
              new_text.append(contractions[word])
          else:
              new_text.append(word)
      text = " ".join(new_text)
  
  # Format words and remove unwanted characters
  text = re.sub(r'&amp;', '', text) 
  text = re.sub(r'0,0', '00', text) 
  text = re.sub(r'[_"\-;%()|.,+&=*%.,!?:#@\[\]]', ' ', text)
  text = re.sub(r'\'', ' ', text)
  text = re.sub(r'\$', ' $ ', text)
  text = re.sub(r'j k ', ' jk ', text)
  text = re.sub(r' s ', ' ', text)
  text = re.sub(r' yr ', ' year ', text)
  text = re.sub(r' l g b t ', ' lgbt ', text)
  if remove_whitespaces:
    text = re.sub(r' ', '', text)
  
  # Optionally, remove stop words
  if remove_stopwords:
      text = text.split()
      stops = set(stopwords.words("english"))
      text = [w for w in text if not w in stops]
      text = " ".join(text)
  return text

from nltk import ngrams
from nltk import word_tokenize, ngrams
from nltk import sent_tokenize, word_tokenize, pos_tag


class CharLevelTextNgram:
  def __init__(self, dataset):
    self.dataset = dataset.copy()
    self.author_profiles = None

  def __preprocess_text_array(self, text_array):
    dataset_len = len(text_array)
    for i in range(dataset_len):
      text = self.__preprocess_text(text_array[i])
      text_array.iloc[i] = text
    return text_array

  def __preprocess_text(self, text):
    return clean_text(text, remove_whitespaces=True)

  def __create_author_profiles(self, dataset, n, L):
    for i in range(len(dataset)):
      dataset[i] = self.__create_author_profile(dataset[i], n, L)
    return dataset

  def __create_author_profile(self, text, n, L):
    cleaned_text = self.__preprocess_text(text)
    df_ngram = ngrams(cleaned_text, n)
    ngram_distribution = nltk.FreqDist(df_ngram)
    normalized_distribution = self.__normalize_author_profile(ngram_distribution)
    return ngram_distribution.most_common(L)

  def __normalize_author_profile(self, frequence_distribution):
    total = frequence_distribution.N()
    for word in frequence_distribution:
      frequence_distribution[word] /= float(total)
    return frequence_distribution

  def __list_to_dict(self, list):
    dictionary = {}
    for index in range(len(list)):
      key = list[index][0]
      dictionary[key] = list[index][1]
    return dictionary

  def __get_union(self, keys1, keys2):
    dictionary = {}
    for key in keys1:
      if key not in dictionary.keys():
        dictionary[key] = True

    for key in keys2:
      if key not in dictionary.keys():
        dictionary[key] = True
    return dictionary

  def __calculate_dissimilarity(value1, value2):
    return ((2*(value1 - value2))/(value1 + value2))**2

  def __find_most_similar_profile(self, training_author_profiles, author_profile):
    similarity_score = sys.maxsize
    most_similar_author = 0
    
    n_profiles = len(self.author_profiles)

    test_profile = self.__list_to_dict(author_profile)
    keys1 = test_profile.keys()
    for i in range(n_profiles):
      author_profile = self.__list_to_dict(self.author_profiles[i])
      keys2 = author_profile.keys()

      keys = self.__get_union(keys1, keys2)

      sum = 0
      for key in keys:
        value1, value2 = 0,0
        try:
          value1 = test_profile[key]
        except Exception:
          pass

        try:
          value2 = author_profile[key]
        except Exception:
          pass

        sum += self.__calculate_dissimilarity(value1, value2)

      if sum < similarity_score:
        similarity_score = sum
        index = i
    return index, self.dataset.index[index]

  def __calculate_dissimilarity(self, value1, value2):
    return ((2*(value1 - value2))/(value1 + value2))**2

  def fit(self, n, L):
    self.n, self.L = n, L
    self.cleaned_dataset = self.__preprocess_text_array(self.dataset)
    self.author_profiles = self.__create_author_profiles(self.cleaned_dataset, n, L)

  def predict(self, text):
    cleaned_text = self.__preprocess_text(text)
    author_profile = self.__create_author_profile(text, self.n, self.L)
    return self.__find_most_similar_profile(self.author_profiles, author_profile)

  def evaluate(self, X, y):
    return True

  def save(self):
    return True


class WordLevelPOSNgram:
  def __init__(self, dataset):
    self.dataset = dataset.copy()
    self.author_profiles = None

  def __preprocess_text_array(self, text_array):
    dataset_len = len(text_array)
    for i in range(dataset_len):
      text = self.__preprocess_text(text_array[i])
      text_array.iloc[i] = text
    return text_array

  def __preprocess_text(self, text):
    return clean_text(text, remove_whitespaces=False)

  def __create_author_profiles(self, dataset, n, L):
    for i in range(len(dataset)):
      dataset[i] = self.__create_author_profile(dataset[i], n, L)
    return dataset

  def __create_author_profile(self, text, n, L):
    cleaned_text = self.__preprocess_text(text)
    pos_tags = pos_tag(word_tokenize(cleaned_text))
    pos_tags = [tag[1] for tag in pos_tags]
    df_ngram = ngrams(pos_tags, n)
    ngram_distribution = nltk.FreqDist(df_ngram)
    normalized_distribution = self.__normalize_author_profile(ngram_distribution)
    return ngram_distribution.most_common(L)

  def __normalize_author_profile(self, frequence_distribution):
    total = frequence_distribution.N()
    for word in frequence_distribution:
      frequence_distribution[word] /= float(total)
    return frequence_distribution

  def __list_to_dict(self, list):
    dictionary = {}
    for index in range(len(list)):
      key = list[index][0]
      dictionary[key] = list[index][1]
    return dictionary

  def __get_union(self, keys1, keys2):
    dictionary = {}
    for key in keys1:
      if key not in dictionary.keys():
        dictionary[key] = True

    for key in keys2:
      if key not in dictionary.keys():
        dictionary[key] = True
    return dictionary

  def __calculate_dissimilarity(value1, value2):
    return ((2*(value1 - value2))/(value1 + value2))**2

  def __find_most_similar_profile(self, training_author_profiles, author_profile):
    similarity_score = sys.maxsize
    most_similar_author = 0
    
    n_profiles = len(self.author_profiles)

    test_profile = self.__list_to_dict(author_profile)
    keys1 = test_profile.keys()
    for i in range(n_profiles):
      author_profile = self.__list_to_dict(self.author_profiles[i])
      keys2 = author_profile.keys()

      keys = self.__get_union(keys1, keys2)

      sum = 0
      for key in keys:
        value1, value2 = 0,0
        try:
          value1 = test_profile[key]
        except Exception:
          pass

        try:
          value2 = author_profile[key]
        except Exception:
          pass

        sum += self.__calculate_dissimilarity(value1, value2)

      if sum < similarity_score:
        similarity_score = sum
        index = i
    return index, self.dataset.index[index]

  def __calculate_dissimilarity(self, value1, value2):
    return ((2*(value1 - value2))/(value1 + value2))**2

  def fit(self, n, L):
    self.n, self.L = n, L
    self.cleaned_dataset = self.__preprocess_text_array(self.dataset)
    self.author_profiles = self.__create_author_profiles(self.cleaned_dataset, n, L)

  def predict(self, text):
    cleaned_text = self.__preprocess_text(text)
    author_profile = self.__create_author_profile(text, self.n, self.L)
    return self.__find_most_similar_profile(self.author_profiles, author_profile)

  def evaluate(self, X, y):
    return True

  def save(self):
    return True


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer

class Direct_BOW_Model:  
  def __init__(self, model):
    self.models = {
        'SVC': LinearSVC(),
        'RF': RandomForestClassifier(),
        'MNB': MultinomialNB()       
    }
    self.model = self.models[model]
    self.bow_transformer = None

  def fit(self, X_train, y_train):
    self.bow_transformer = CountVectorizer().fit(X_train)
    text_bow_train = self.bow_transformer.transform(X_train)
    self.model.fit(text_bow_train, y_train)

  def predict(self, text):
    return self.model.predict(self.bow_transformer.transform([clean_text(text, remove_whitespaces=False)]))[0]

  def evaluate(self, X, y):
    text_bow_X = self.bow_transformer.transform(X)
    return self.model.score(text_bow_X, y)



class TfIdf_BOW_Model:  
  def __init__(self, model):
    self.models = {
        'SVC': LinearSVC(),
        'RF': RandomForestClassifier(),
        'MNB': MultinomialNB()       
    }
    self.model = text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', self.models[model] ),
    ])

  def fit(self, X_train, y_train):
    self.model.fit(X_train, y_train)

  def predict(self, text):
    return self.model.predict([clean_text(text, remove_whitespaces=False)])[0]

  def evaluate(self, X, y):
    predicted = self.model.predict(X)
    return classification_report(y, predicted)
    
    
print('I am here-10')