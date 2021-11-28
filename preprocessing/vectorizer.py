import numpy as np
import sys

current_module = sys.modules[__name__]

from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer




class Vectorizer(BaseEstimator):

    def __init__(self, vectorizer: int, ngram_range=(1, 1), max_features=None) -> None:
        self.ngram_range = ngram_range
        self.max_features = max_features

        if vectorizer == 1:
            self.vect = CountVectorizer(ngram_range=ngram_range, max_features=max_features)
        elif vectorizer == 2:
            self.vect = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)
        else:
            print('The vectorizer id selected doesn\'t belong to any vectorizer')
    
    def fit(self, X, y=None):

        return self.vect.fit(X, y)

    def transform(self, X, y=None):
        
        return self.vect.transform(X)
    
    def fit_transform(self, X, y=None):

        return self.vect.fit_transform(X, y)
