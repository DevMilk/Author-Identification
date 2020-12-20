from flask import Flask, jsonify, redirect, request
from flask import render_template
import os
from enum import Enum
try:
    import cPickle as pickle
except:
    import pickle
import pandas as pd
from flasgger import Swagger
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
swagger = Swagger(app)
class Model(Enum):
    BOW = 2
    NGRAM = 1
    STYLE = 3
    ALL = 0

@app.route('/')
def hello():
    return render_template("index.html");


@app.route('/predict',methods= ["POST"])
def predict(): #Model.NGRAM şeklinde bir parametre
    predictions = []
    parameters = request.get_json()
    
    text = parameters.get("text")
    modelEnum = parameters.get("ModelEnum")
    args = parameters.get("args")


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