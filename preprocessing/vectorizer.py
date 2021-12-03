import sys

current_module = sys.modules[__name__]

from sklearn.base import BaseEstimator
from typing import Tuple
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer




class Vectorizer(BaseEstimator):
    """This class is used to be a text vectorizer. Handle count vectorization and TF-IDF vectorization.

    Args:
        BaseEstimator ([type]): inherit from scikit-learn BaseEstimator to fit in pipeline properly.
    """

    def __init__(self, vectorizer: int, ngram_range=(1, 1), max_features:int=None) -> None:
        """ init vectorizer.

        Args:
            vectorizer (int): Choose Count or TF-IDF.
            ngram_range (tuple, optional): ngram wanted. Defaults to (1, 1).
            max_features (int, optional): max features to use, correspond to dimensions. Defaults to None.
        """
        self.ngram_range = ngram_range
        self.max_features = max_features

        if vectorizer == 1:
            self.vectorizer = CountVectorizer(ngram_range=ngram_range, max_features=max_features)
        elif vectorizer == 2:
            self.vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)
        else:
            print('The vectorizer id selected doesn\'t belong to any vectorizer')
    
    def fit(self, X, y=None):
        """fit method.

        Args:
            X ([type]): The training data.
            y ([type], optional): The training targets. Defaults to None.

        Returns:
            [type]: The bottom level vectorizer (Count or Tfidf from scikit-learn).
        """

        return self.vectorizer.fit(X, y)

    def transform(self, X, y=None):
        """Transform method.

        Args:
            X ([type]): The data to be transformed.
            y ([type], optional): Targets. Defaults to None.

        Returns:
            [type]: The data transformed.
        """
        return self.vectorizer.transform(X)
    
    def fit_transform(self, X, y=None):
        """fit and transformed.

        Args:
            X ([type]): The data to fit the transform.
            y ([type], optional): Targets. Defaults to None.

        Returns:
            [type]: the data transformed.
        """

        return self.vectorizer.fit_transform(X, y)

    def update_params(self, ngram_range=(1,1), max_features=None):
        """To updates parameters of the bottom level vectorizer.

        Args:
            ngram_range (tuple, optional): ngrams. Defaults to (1,1).
            max_features ([type], optional): maximum features. Defaults to None.
        """
        self.ngram_range = ngram_range
        self.max_features = max_features

        self.vectorizer.ngram_range = ngram_range
        self.vectorizer.max_features = max_features
