import re
import nltk
import string
import seaborn as sns
import pandas as pd
import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from models.model_getter import getStyleBased
nltk.download('punkt')
nltk.download('stopwords')

stopword = set(stopwords.words('english'))
stopword_af = stopword
nltk.download('averaged_perceptron_tagger')
ps=PorterStemmer()

class StyleBased:
    def __init__(self, subModel):
        self.model = getStyleBased(subModel)
    
    def __extract_features(self, txt):
        #Feat 1: total number of sentences
        total_sents = len(nltk.sent_tokenize(txt))
        #Feat 2: total number of words (raw)
        total_tokens_raw = len(nltk.word_tokenize(txt))
        #Feat 3:total no. of tokens (words) when all words in lower case
        total_tokens_lower = len(nltk.word_tokenize(txt.lower()))
        # Feat 4: Total unique no; of words in lower case
        unique_tokens_lower = len(set(nltk.word_tokenize(txt.lower())))
        # Feat 5:Total unique no; of tokens in lower case & stemmed
        unique_tokens_lower_stem = len(set([ps.stem(w) for w in nltk.word_tokenize(txt.lower())]))
        # Feat 6: Total unique no; of tokens in lower case + stemmed - stopwords
        tokens_lower_stem_sword= len(set([ps.stem(w) for w in nltk.word_tokenize(txt.lower()) if ps.stem(w) not in stopword_af]))
        # Feat 7: total counts of all stopwords (duplicate also considered)
        stopw_count=len([ps.stem(w) for w in nltk.word_tokenize(txt.lower()) if w in stopword_af ])
        # Feat 8:Total no; of punctuations 
        total_puncts = len([char for char in txt if char in string.punctuation])
        # Feat 9:Total nouns in lower+stemmed tokens
        lower_stem_ls=[ps.stem(w) for w in nltk.word_tokenize(txt.lower())]
        nouns=len([i[0] for i in nltk.pos_tag(lower_stem_ls) if i[1] in ['NN','NNS','NNP','NNPS']])
        #Feat 10:Total verbs in lower+stemmed tokens
        verbs=len([i[0] for i in nltk.pos_tag(lower_stem_ls) if i[1] in ['VB','VBD','VBG','VBN','VBP','VBZ']])
        #Feat 11:Average words per sentence
        avg_words=len(re.split('[\s]',txt))/len(nltk.sent_tokenize(txt))
        #Feat 12:Average words length (lower case + stemmed - stop words)
        lower_sword_ls = [word for word in (nltk.word_tokenize(txt.lower())) if word not in stopword_af]
        avg_word_len=sum([len(w) for w in lower_sword_ls])/len(lower_sword_ls)
        #Feat 13 :Sentiment score 
        lower_stem_sword=[ps.stem(w) for w in nltk.word_tokenize(txt.lower()) if w not in stopword_af]
        return {
            'total_sents': total_sents,
            'total_tokens_raw': total_tokens_raw,
            'total_tokens_lower': total_tokens_lower,
            'unique_tokens_lower': unique_tokens_lower,
            'unique_tokens_lower_stem': unique_tokens_lower_stem,
            'tokens_lower_stem_sword':tokens_lower_stem_sword,
            'stopw_count':stopw_count,
            'total_puncts':total_puncts,
            'nouns':nouns,
            'verbs':verbs,
            'avg_words':avg_words,
            'avg_word_len':avg_word_len
        }
        
    def predict(self, txt):
        features = self.__extract_features(txt)
        auth_df = pd.DataFrame([features])
        auth_df.drop(['total_tokens_raw','total_tokens_lower'],axis=1,inplace=True)
        auth_df.drop(['tokens_lower_stem_sword'],axis=1,inplace=True)
        return self.model.predict(auth_df)[0]