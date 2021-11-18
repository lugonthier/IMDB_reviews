import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

def generate_holdout(generate_kfold:bool=True, k=5):
    """Generate 3 datasets. The train set, the validation set and the test set.
    if generate_kflod == True, generate k fold based on the train set. Then we saved the indices into the train set. 
    

    Args:
        generate_kfold (bool, optional): to decide if we generate k folds. Defaults to True.
        k (int, optional): number of fold. Defaults to 5.
    """
    df = pd.read_csv('data/IMDB_Dataset.csv')
    
    df_origin_train, df_test = train_test_split(df, test_size=.2, random_state=42)

    df_train, df_valid = train_test_split(df_origin_train, test_size=.2, random_state=42)
    print(df_train.head())
    if generate_kfold:
        df_train = K_fold(df_train, k)

    df_train.to_csv('data/IMDB_train.csv', index=False)
    df_valid.to_csv('data/IMDB_valid.csv', index=False)
    df_test.to_csv('data/IMDB_test.csv', index=False)



def K_fold(df, k:int):
    """fold generation using stratified kfold to balance each fold.

    Args:
        df ([type]): dataframe
        k (int): number of folder

    Returns:
        [type]: dataframe with k new column with train or test as possible values.
    """
    skf = StratifiedKFold(n_splits=k, random_state=None, shuffle=True)
        
    indices = skf.split(df.review,df.sentiment)
    
    
    for train, test in indices:
        
        df['fold_' + str(k)] = ['train' if (index in train) else 'test' for index in range(len(np.concatenate((train,test)))) ]
        k -=1
        
    return df

if __name__ == "__main__":
    generate_holdout()