import sys

current_module = sys.modules[__name__]

from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer




class Vectorizer(BaseEstimator):

    def __init__(self, vectorizer: int, ngram_range=(1, 1), max_features=None) -> None:
        self.ngram_range = ngram_range
        self.max_features = max_features

        if vectorizer == 1:
            self.vectorizer = CountVectorizer(ngram_range=ngram_range, max_features=max_features)
        elif vectorizer == 2:
            self.vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)
        else:
            print('The vectorizer id selected doesn\'t belong to any vectorizer')
    
    def fit(self, X, y=None):

        return self.vectorizer.fit(X, y)

    def transform(self, X, y=None):
        
        return self.vectorizer.transform(X)
    
    def fit_transform(self, X, y=None):

        return self.vectorizer.fit_transform(X, y)

    def update_params(self, ngram_range=(1,1), max_features=None):
        self.ngram_range = ngram_range
        self.max_features = max_features

        self.vectorizer.ngram_range = ngram_range
        self.vectorizer.max_features = max_features
