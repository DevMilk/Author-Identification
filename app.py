from flask import Flask, jsonify, redirect, request
from flask import render_template
import os
from enum import Enum
import pickle
import pandas as pd
from flasgger import Swagger

from models.model_getter import *


Model_dict = {
    "NGRAM": {
        "WRD": get_ngram_wrd(),
        "CHR": get_ngram_chr(),
        "POS": get_ngram_pos()
    },
    "BASIC-BOW": {
        "RF": get_basic_bow_rf(),
        "MNB": get_basic_bow_mnb(),
        "SVC": get_basic_bow_svc()
    },
    "TF-IDF-BOW": {
        "RF": get_tf_idf_bow_rf(),
        "MNB": get_tf_idf_bow_rf(),
        "SVC": get_tf_idf_bow_rf()
    },
    "STYLE-BASED": {
        "RF": get_style_based_rf(),
        "SVC": get_style_based_svc()
    }
}

# Model_dict = {
#     "NGRAM": {
#         "WRD": get_ngram_wrd(),
#         "CHR": get_ngram_chr(),
#         # "POS": get_ngram_pos()
#     },
#     "BASIC-BOW": {
#         "RF": None,
#         "MNB": None,
#         "SVC": None
#     },
#     "TF-IDF-BOW": {
#         "RF": None,
#         "MNB": None,
#         "SVC": None
#     },
#     "STYLE-BASED": {
#         "RF": None,
#         "SVC": None
#     }
# }



# def load_model(mainModel, subModel):
#   pickle_filename = "./models/saved_models/"+mainModel+ '/' + subModel+".pkl"
#   picklefile = open(pickle_filename, 'rb')
#   model = pickle.load(picklefile)
#   picklefile.close()
#   return model


# def load_models():
#     models = []
#     try:
#         for mainModel in list(Model_dict.keys()):
#             if mainModel == 'NGRAM': continue
#             for subModel in list(Model_dict[mainModel].keys()):
#                 model= mainModel + '/' + subModel
#                 try:
#                     Model_dict[mainModel][subModel] = load_model(mainModel, subModel)
#                     print(model + 'was installed successfully!')
#                 except Exception as e:
#                     print('An error has occurred while installing ' + model)
#                     print(e)
                
#     except Exception as e: 
#         print(e)
#     return models


def load_dataset():
    col_list = ["author","text"]
    return pd.read_csv("dataset/train.csv",usecols=col_list),pd.read_csv("dataset/test.csv",usecols=col_list)

# load_models()

train_data, test_data = load_dataset()

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


def runMethodOfModel(methodName, args,material):
    results = []
    requestedModel = getModelFromDict(args)
    if(args[-1]=="ALL"):
        for key in list(Model_dict.keys()):
            results.append(Model_dict[key].getattr(requestedModel, methodName)(material,[]))
    else:
        model, last_args =  getModelFromDict(args)
        results.append(getattr(model,methodName)(material,last_args))

    return results

@app.route('/')
def hello():
    return render_template("index.html");


#Modele tahmin yaptırır

@app.route('/predict', methods= ["POST"])
def predict():
    parameters = request.get_json()
    text = parameters.get("text")
    args = parameters.get("args")

    return runMethodOfModel("predict",args,text)



#Mevcut modeli eğitir ve train accuracy'sini döndürür

@app.route('/train', methods= ["POST"])
def train(): 
    parameters = request.get_json()
    args = parameters.get("args")
    return runMethodOfModel("train",args,test_data)



#Mevcut modeli test eder ve test accuracy'sini döndürür

@app.route('/test', methods= ["POST"])
def test(): 
    parameters = request.get_json()
    args = parameters.get("args")
    return runMethodOfModel("test", args, train_data)


if __name__ == '__main__':
    pass