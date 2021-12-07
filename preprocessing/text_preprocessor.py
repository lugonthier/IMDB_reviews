import re
import string
import numpy as np
import pandas as pd
import nltk
nltk.download('stopwords')
import sys

current_module = sys.modules[__name__]

from typing import List, Union
from sklearn.base import BaseEstimator
from utils.utils import transform_to_ndarray

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer


class TextPreprocessor(BaseEstimator):
    """Class used to perform text preprocessing in order to give it to a pipeline.

    Args:
        stopwords : List | str
            If is a list, then it will represent all stopwords that must be removed. 
            If is a string, it should be a language present in nltk stopwords.
        
        normalization : int 
            The normalization that will be perfomed.
            if 1 => WordNetLemmatizer is used.
            if 2 => PorterStemmer is used.
            if none of the cases mentioned above, the data aren't normalized.
        
    """

    def __init__(self, stopwords:Union[List,str] ='english', normalization=None):
        
        if type(stopwords) == str:
            self.stopwords = set(nltk.corpus.stopwords.words(stopwords))
        else:
            self.stopwords = stopwords

        self.normalization = normalization

    def fit(self, X: Union[List, np.ndarray, pd.Series], y: Union[List, np.ndarray, pd.Series]=None) -> None:
        """This method doesn't perform any action for now. Just here to allow the class in a sklearn pipeline.

        Args:
            X (Union[List, np.ndarray, pd.Series]): Data that must be processed.
            y (Union[List, np.ndarray, pd.Series], optional): Targets corresponding to X. Defaults to None.
        """
        pass
        

    def transform(self, X: Union[List, np.ndarray, pd.Series], y: Union[List, np.ndarray, pd.Series]=None) -> np.ndarray:
        """This method processed all data in X, respecting the user's choices.

        Args:
            X (Union[List, np.ndarray, pd.Series]): Data that must be processed.
            y (Union[List, np.ndarray, pd.Series], optional): Targets corresponding to X. Defaults to None.

        Returns:
            np.ndarray: Return the processed data in X.
        """
        X = transform_to_ndarray(X)
        y = transform_to_ndarray(y)
        
        vect_punctuation = np.vectorize(self.remove_punctuation)
        X_transformed = vect_punctuation(X)
        
        vect_token = np.vectorize(self.tokenize)
        X_transformed = vect_token(X_transformed)
        
        vect_stopwords = np.vectorize(self.remove_stopwords)
        X_transformed = vect_stopwords(X_transformed)

        vect_normalize = np.vectorize(self.normalize)
        X_transformed = vect_normalize(X_transformed)

        X_transformed = [" ".join(Xi) for Xi in X_transformed]

        return X_transformed


    def fit_transform(self, X: Union[List, np.ndarray, pd.Series], y: Union[List, np.ndarray, pd.Series] = None) -> np.ndarray:
        """Equivalent to calling the fit method followed by the transform method.

        Args:
            X (Union[List, np.ndarray, pd.Series]): Data that must be processed.
            y (Union[List, np.ndarray, pd.Series], optional): Targets corresponding to X. Defaults to None.

        Returns:
            np.ndarray: Return the processed data in X.
        """
        self.fit(X, y)
        
        return self.transform(X, y)
    


    def remove_punctuation(self, text: str) -> str:
        """This method remove punctuation of a text. (and lower all char)

        Args:
            text (str): The text that must be processed.

        Returns:
            str: The text without punctuation.
        """
        regex = re.compile('[%s]' % re.escape(string.punctuation))
        text = text.lower()
        return regex.sub('', text)



    def remove_special_str(self, text: str, str: str) -> str:
        """Remove a string from another string.

        Args:
            text (str): The string that must be processed. 
            str (str): The string that must be removed.

        Returns:
            str: The string processed.
        """
        return str.sub('', text)



    def remove_stopwords(self, words: List) -> Union[List, np.ndarray]:
        """Method to remove all stopwords in a list of words. Based on the stopwords class attribut. 

        Args:
            words (List): A list of words to be processed.

        Returns:
            Union(List, np.ndarray): List of words without the stop words.
        """
        if len(self.stopwords)>0:
            return np.array([word  for word in words if word not in self.stopwords], dtype=object)
        else:
            return words



    def tokenize(self, text: str) -> np.ndarray:
        """Transform a text in an array of words.

        Args:
            text (str): The text that must be processed.

        Returns:
            np.ndarray: The array of words.
        """
        return np.array(text.split(), dtype=object)



    def normalize(self, X:np.ndarray) -> np.ndarray:
        """Normalize the text with lemmatizer or stemmer. It depends on the "normalization" class attribut.

        Args:
            X (np.ndarray): Data that must be processed.

        Returns:
            np.ndarray: Data normalized.
        """
        if self.normalization == 1:
            normalizer = WordNetLemmatizer()
            return np.array([normalizer.lemmatize(text) for text in X], dtype=object)            
        
        elif self.normalization == 2:
            normalizer = PorterStemmer()
            return np.array([normalizer.stem(text) for text in X], dtype=object)
        
        else:
            return X

    