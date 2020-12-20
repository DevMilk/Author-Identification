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

def load_models():
    models = []
    try:
        for mainModel in list(Model_dict.keys()):
            for subModel in list(Model_dict[mainModel].keys()):
                Model_dict[mainModel][subModel] =pickle.load(open("models/"+mainModel+"/"+subModel+".pkl","rb",-1))
    except: 
        pass
    return models


def load_dataset():
    col_list = ["author","text"]
    return pd.read_csv("dataset/train.csv",usecols=col_list),pd.read_csv("dataset/test.csv",usecols=col_list)


models = load_models() #Modelleri localden çekecek
train_data,test_data = load_dataset()

app = Flask(__name__)
swagger = Swagger(app)





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


def runMethodOfModel(methodName,args,material):
    results = []
    if(args[-1]=="ALL"):
        for key in list(requestedModel.keys()):
            results.append(Model_dict[key].getattr(model,methodName)(material,[]))
    else:
        model, last_args =  getModelFromDict(args)
        results.append(getattr(model,methodName)(material,last_args))

    return results



@app.route('/')
def hello():
    return render_template("index.html");


#Modele tahmin yaptırır

@app.route('/predict',methods= ["POST"])
def predict():
    parameters = request.get_json()
    text = parameters.get("text")
    args = parameters.get("args")

    return runMethodOfModel("predict",args,text)



#Mevcut modeli eğitir ve train accuracy'sini döndürür

@app.route('/train')
def train(): 
    parameters = request.get_json()
    args = parameters.get("args")
    return runMethodOfModel("train",args,test_data)



#Mevcut modeli test eder ve test accuracy'sini döndürür

@app.route('/test')
def test(): 
    parameters = request.get_json()
    args = parameters.get("args")
    return runMethodOfModel("test",args,train_data)


if __name__ == '__main__':
    app.run()