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


Model_dict = {
    "NGRAM": {
        "WRD": None ,
        "CHR": None,
        "POS": None
    },
    "BOW": {
        "BASIC": None,
        "TF-IDF": None,
    }
}


def getModelFromDict(args):
    print(args[-1])
    requestedModel = Model_dict[args[-1]]
    counter = len(args)-2
    while(isinstance(requestedModel,dict)):
        if(counter<0):
            print(list(requestedModel.keys())[0])
            requestedModel =requestedModel[list(requestedModel.keys())[0]]
        else:
            requestedModel = requestedModel[args[counter]]
            print(args[counter])
            counter = counter -1
    counter = counter +1 
    last_args = args[:counter]
    if(counter<0):
        last_args = []
    print(last_args)
    return requestedModel, last_args



@app.route('/')
def hello():
    return render_template("index.html");


@app.route('/predict',methods= ["POST"])
def predict(): #Model.NGRAM şeklinde bir parametre
    predictions = []
    parameters = request.get_json()
    
    text = parameters.get("text")
    args = parameters.get("args")
    print(args)
    if(args[-1]=="ALL"):
        for i in range(0,3):
            predictions.append(models[i].predict(text))
    else:
        model, last_args =  getModelFromDict(args)
        predictions.append(model.predict(text,last_args))

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