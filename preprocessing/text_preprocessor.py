import re
import string
import numpy as np
import pandas as pd

from typing import List, Union
from sklearn.base import BaseEstimator

class TextPreprocessor(BaseEstimator):
    
    def __init__(self, stopwords:List = [], normalization=None):
        
        self.stopwords = stopwords
        self.normalization = normalization

    def fit(self, X: Union[List, np.ndarray, pd.Series], y: Union[List, np.ndarray, pd.Series]) -> None:
        self.X = self.__transform_type(X)
        self.y = self.__transform_type(y)

        

    def transform(self, X: Union[List, np.ndarray, pd.Series], y: Union[List, np.ndarray, pd.Series]) -> np.ndarray:
        
        pass

    def fit_transform(self, X: Union[List, np.ndarray, pd.Series], y: Union[List, np.ndarray, pd.Series]) -> np.ndarray:
        
        self.fit(X,y)
        
        return self.transform(X, y)
    
    def remove_punctuation(self, text: str) -> str:
        regex = re.compile('[%s]' % re.escape(string.punctuation))
        text = text.lower()
        return regex.sub('', text)

    def remove_special_str(self, text: str, str: str) -> str:
        
        return str.sub('', text)

    def remove_stopwords(self, words: List) -> List:

        if len(self.stopwords > 0):
            return [word  for word in words if word not in self.stopwords]
        else:
            return words

    def tokenize(self, text: str) -> List:
        
        return text.split()

    def normalize(self):
        pass

    def __transform_type(self, data):
        
        if type(data) == List:
            return np.array(data, dtype=object)

        elif (type(data) == pd.Series):
            return data.to_numpy()

        elif (type(data) == np.ndarray):
            return data