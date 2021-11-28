import numpy as np
import sys

current_module = sys.modules[__name__]


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer




def select_vectorizer(vect_selected: int) -> any:
    """This function return a Vectorizer based on parameter.

    Args:
        vect_selected (int): 
            1 => CountVectorizer
            2 => TfidfVectorizer

    Returns:
        any: A Vectorizer
    """
    if vect_selected == 1:
        return CountVectorizer( max_features=12000)

    elif vect_selected == 2:
        return TfidfVectorizer( )

