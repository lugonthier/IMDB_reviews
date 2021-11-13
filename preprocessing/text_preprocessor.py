import re
import string
import numpy as np
import pandas as pd
import nltk
import sys

current_module = sys.modules[__name__]

from typing import List, Union
from sklearn.base import BaseEstimator
from utils.utils import transform_to_ndarray

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer


class TextPreprocessor(BaseEstimator):
    
    def __init__(self, stopwords:List = [], normalization=None):
        
        self.stopwords = stopwords
        self.normalization = normalization

    def fit(self, X: Union[List, np.ndarray, pd.Series], y: Union[List, np.ndarray, pd.Series]) -> None:
        
        pass
        

    def transform(self, X: Union[List, np.ndarray, pd.Series], y: Union[List, np.ndarray, pd.Series]) -> np.ndarray:
        
        X = transform_to_ndarray(X)
        y = transform_to_ndarray(y)

        vect_punctuation = np.vectorize(self.remove_punctuation)
        X_transformed = vect_punctuation(X)

        #vect_special_str = np.vectorize(self.remove_special_str)
        vect_token = np.vectorize(self.tokenize)
        X_transformed = vect_token(X_transformed)

        vect_stopwords = np.vectorize(self.remove_stopwords)
        X_transformed = vect_stopwords(X_transformed)

        vect_normalize = np.vectorize(self.normalize)
        X_transformed = vect_normalize(X_transformed)


        return X_transformed, y


    def fit_transform(self, X: Union[List, np.ndarray, pd.Series], y: Union[List, np.ndarray, pd.Series]) -> np.ndarray:
        
        self.fit(X, y)
        
        return self.transform(X, y)
    


    def remove_punctuation(self, text: str) -> str:
        
        regex = re.compile('[%s]' % re.escape(string.punctuation))
        text = text.lower()
        return regex.sub('', text)



    def remove_special_str(self, text: str, str: str) -> str:
        
        return str.sub('', text)



    def remove_stopwords(self, words: List) -> List:

        if len(self.stopwords)>0:
            return np.array([word  for word in words if word not in self.stopwords], dtype=object)
        else:
            return words



    def tokenize(self, text: str) -> List:
        
        return np.array(text.split(), dtype=object)



    def normalize(self, X):
        
        if self.normalization == 1:
            normalizer = WordNetLemmatizer()
            return np.array([normalizer.lemmatize(text) for text in X], dtype=object)            
        
        elif self.normalization == 2:
            normalizer = PorterStemmer()
            return np.array([normalizer.stem(text) for text in X], dtype=object)
        
        else:
            return X

    