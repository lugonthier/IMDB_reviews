import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

def generate_holdout(generate_kfold:bool=True, k=5):
    df = pd.read_csv('/Users/gonthierlucas/Desktop/DS_project/IMDB_reviews/data/IMDB_Dataset.csv')
    
    df_origin_train, df_test = train_test_split(df, test_size=.2, random_state=42)

    df_train, df_valid = train_test_split(df_origin_train, test_size=.2, random_state=42)
    
    if generate_kfold:
        df_train = K_fold(df_train, k)

    df_train.to_csv('/Users/gonthierlucas/Desktop/DS_project/IMDB_reviews/data/IMDB_train.csv', index=False)
    df_valid.to_csv('/Users/gonthierlucas/Desktop/DS_project/IMDB_reviews/data/IMDB_valid.csv', index=False)
    df_test.to_csv('/Users/gonthierlucas/Desktop/DS_project/IMDB_reviews/data/IMDB_test.csv', index=False)



def K_fold(df, k):
    skf = StratifiedKFold(n_splits=k, random_state=None, shuffle=True)
        
    indices = skf.split(df.review,df.sentiment)
    
    
    for train, test in indices:
        
        df['fold_' + str(k)] = ['train' if (index in train) else 'test' for index in range(len(np.concatenate((train,test)))) ]
        k -=1
        
    
if __name__ == "__main__":
    generate_holdout()