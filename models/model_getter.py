from models.preprocessing import clean_text
from models.dataset import get_merged_dataset
from models.dataset import get_train_test_dataset
from models.ngram_models import CharLevelTextNgram
from models.ngram_models import WordLevelPOSNgram
from models.ngram_models import WordLevelTextgram
from models.basic_bow_models import Direct_BOW_Model
from models.basic_bow_models import TfIdf_BOW_Model
from utils.model import load_model


merged_dataset = get_merged_dataset()
X_train, y_train, X_test, y_test = get_train_test_dataset()

# NGRAM MODELS
def getModel(model,init_arg,fit_args):
    model_object = model(init_arg)
    model_object.fit(*fit_args)
    print('{} was installed successfully!'.format(model))
    return model_object

def getNgram(ngramClass):
    return getModel(ngramClass,merged_dataset,(3,30))

#Basıc Bow Model
def getBasicBow(ml):
    return getModel(Direct_BOW_Model,ml,(X_train,y_train))

# TF-IDF BOW MODELS
def getTfidfBow(ml):
    return getModel(TfIdf_BOW_Model,ml,(X_train,y_train))


# STYLE-BASED MODELS
def getStyleBased(subModel):
    return load_model('STYLE-BASED', subModel);

