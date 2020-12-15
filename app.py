from flask import Flask
from flask import render_template
import os
from enum import Enum
try:
    import cPickle as pickle
except:
    import pickle
import pandas as pd


def load_models():
    models = []
    for modelName in os.listdir("models"):
        models.append(pickle.load(open(modelName,"rb",-1)))
    return models


def load_dataset():
    col_list = ["author","text"]
    return pd.read_csv("dataset/train.csv",usecols=col_list),pd.read_csv("dataset/test.csv",usecols=col_list)


models = []
models = load_models() #Modelleri localden çekecek
train,test = load_dataset()

app = Flask(__name__)

class Model(Enum):
    BOW = 0
    NGRAM = 2
    STYLE = 3
    ALL = 1


@app.route('/')
def hello():
    return render_template("index.html");


@app.route('/predict')
def predict(text,modelEnum,**args): #Model.NGRAM şeklinde bir parametre
    predictions = []
    if(modelEnum==Model.ALL):
        for i in range(0,3):
            predictions.append(models[i].predict(text,{"Additional":"ALL"}))
    else:
        predictions.append(models[modelEnum].predict(text,args))

    return predictions

#Mevcut modeli eğitir ve train accuracy'sini döndürür
@app.route('/train')
def train(modelEnum,**args): 
    return models[modelEnum].train(train,args)

#Mevcut modeli test eder ve test accuracy'sini döndürür
@app.route('/test')
def test(modelEnum,**args): 
    return models[modelEnum].test(test,args)


if __name__ == '__main__':
    app.run()