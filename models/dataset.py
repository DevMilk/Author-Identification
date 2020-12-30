import os
import sys
import pandas as pd
import numpy as np
from models.preprocessing import clean_text

training_path = './dataset/train.csv'
test_path = './dataset/test.csv'
df_dataset = pd.read_csv(training_path)

# Grouped dataset
def get_grouped_dataset():
    grouped_dataset = df_dataset.groupby("author")["text"].apply(list)
    return grouped_dataset

# Merged dataset
def combine_elements(arr):
  for i in range(len(arr)):
    corpora = ''
    for corpus in arr:
      corpora += corpus
  return corpora

def get_merged_dataset():
    merged_dataset = get_grouped_dataset().copy()
    for i in range(merged_dataset.size):
        merged_dataset[i] = combine_elements(merged_dataset[i])
    return merged_dataset

def get_train_test_dataset():
    dfTrain = pd.read_csv(training_path)
    dfTest = pd.read_csv(test_path)

    dfTrain["text"] = dfTrain["text"].apply(lambda x: clean_text(x, remove_whitespaces=False))
    dfTest["text"] = dfTest["text"].apply(lambda x: clean_text(x, remove_whitespaces=False))

    X_train, y_train = dfTrain["text"], dfTrain["author"]
    X_test, y_test = dfTest["text"], dfTest["author"]
    
    return X_train, y_train, X_test, y_test