
import numpy as np
import pandas as pd

from typing import List, Union

def transform_to_ndarray(data):
        
    if type(data) == List:
        return np.array(data, dtype=object)

    elif (type(data) == pd.Series):
        return data.to_numpy()

    elif (type(data) == np.ndarray):
        return data