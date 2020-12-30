from models.preprocessing import clean_text
from models.dataset import get_merged_dataset
from models.dataset import get_train_test_dataset
from models.ngram_models import CharLevelTextNgram
from models.ngram_models import WordLevelPOSNgram
from models.basic_bow_models import Direct_BOW_Model
from models.basic_bow_models import TfIdf_BOW_Model

merged_dataset = get_merged_dataset()
X_train, y_train, X_test, y_test = get_train_test_dataset()

# NGRAM MODELS
def get_ngram_wrd():
    return None

def get_ngram_chr():
    charLevelTextNgram = CharLevelTextNgram(dataset=merged_dataset)
    charLevelTextNgram.fit(n=3, L=30)
    print('ngram_chr was installed successfully!')
    return charLevelTextNgram

def get_ngram_pos():
    wordLevelPOSNgram = WordLevelPOSNgram(dataset=merged_dataset)
    wordLevelPOSNgram.fit(n=3, L=30)
    print('ngram_pos was installed successfully!')
    return wordLevelPOSNgram

# BASIC BOW MODELS
def get_basic_bow_rf():
    direct_bow_model_randomForest = Direct_BOW_Model('RF')
    direct_bow_model_randomForest.fit(X_train, y_train)
    print('basic_bow_rf was installed successfully!')
    return direct_bow_model_randomForest

def get_basic_bow_mnb():
    direct_bow_model_multinomialNB = Direct_BOW_Model('MNB')
    direct_bow_model_multinomialNB.fit(X_train, y_train)
    print('basic_bow_mnb was installed successfully!')
    return direct_bow_model_multinomialNB

def get_basic_bow_svc():
    direct_bow_model_linearSVC = Direct_BOW_Model('SVC')
    direct_bow_model_linearSVC.fit(X_train, y_train)
    print('basic_bow_svc was installed successfully!')
    return direct_bow_model_linearSVC

# TF-IDF BOW MODELS
def get_tf_idf_bow_rf():
    tf_idf_bow_model_randomForest = TfIdf_BOW_Model('RF')
    tf_idf_bow_model_randomForest.fit(X_train, y_train)
    print('tf_idf_bow_rf was installed successfully!')
    return tf_idf_bow_model_randomForest

def get_tf_idf_bow_mnb():
    tf_idf_model_multinomialNB = TfIdf_BOW_Model('MNB')
    tf_idf_model_multinomialNB.fit(X_train, y_train)
    print('tf_idf_bow_mnb was installed successfully!')
    return tf_idf_model_multinomialNB
    
def get_tf_idf_bow_svc():
    tf_idf_model_model_linearSVC = TfIdf_BOW_Model('SVC')
    tf_idf_model_model_linearSVC.fit(X_train, y_train)
    print('tf_idf_bow_svc was installed successfully!')
    return tf_idf_model_model_linearSVC


# STYLE-BASED MODELS
def get_style_based_rf():
    return None

def get_style_based_svc():
    return None

