import nltk
import sys
from nltk import ngrams
from nltk import word_tokenize, ngrams
from nltk import sent_tokenize, word_tokenize, pos_tag
from models.preprocessing import clean_text

#TODO: TEST KISMINI YAP

DEFAULT_N = 3
DEFAULT_L = 30
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

  def __create_author_profile(self, text, n=DEFAULT_N, L=DEFAULT_L):
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
    return self.dataset.index[index]

  def __calculate_dissimilarity(self, value1, value2):
    return ((2*(value1 - value2))/(value1 + value2))**2

  def fit(self, n=DEFAULT_N, L=DEFAULT_L):
    self.n, self.L = n, L
    self.cleaned_dataset = self.__preprocess_text_array(self.dataset)
    self.author_profiles = self.__create_author_profiles(self.cleaned_dataset, n, L)

  def predict(self, text):
    cleaned_text = self.__preprocess_text(text)
    author_profile = self.__create_author_profile(text, self.n, self.L)
    return self.__find_most_similar_profile(self.author_profiles, author_profile)

  def test(self, X, y):
    return True

  def save(self):
    return True
  
class WordLevelTextgram:
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
    words = word_tokenize(cleaned_text)
    words = [word for word in words]
    df_ngram = ngrams(words, n)
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
    return self.dataset.index[index]

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

  def evaluate(self, X_test, y_test):
    y_predicted = []
    for i, x_test in enumerate(X_test):
      y_predicted.append(self.predict(x_test))
    return self.accuracy(y_test, y_predicted)
    
  def accuracy(self, y_test, y_predicted):
    n_test = len(y_test)
    n_true = 0
    for i in range(n_test):
      if y_test[i] == y_predicted[i]:
        n_true += 1
    accuracy = 0
    try:
      accuracy = n_test / n_true
    except Exception:
      pass
    return accuracy

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

  def __create_author_profile(self, text, n=DEFAULT_N, L=DEFAULT_L):
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
    return self.dataset.index[index]

  def __calculate_dissimilarity(self, value1, value2):
    return ((2*(value1 - value2))/(value1 + value2))**2

  def fit(self, n=DEFAULT_N, L=DEFAULT_L):
    self.n, self.L = n, L
    self.cleaned_dataset = self.__preprocess_text_array(self.dataset)
    self.author_profiles = self.__create_author_profiles(self.cleaned_dataset, n, L)

  def predict(self, text):
    cleaned_text = self.__preprocess_text(text)
    author_profile = self.__create_author_profile(text, self.n, self.L)
    return self.__find_most_similar_profile(self.author_profiles, author_profile)

  def test(self, X, y):
    return True

  def save(self):
    return True

