import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    df = pd.read_csv('/Users/gonthierlucas/Desktop/DS_project/IMDB_reviews/data/IMDB_Dataset.csv')

    df_origin_train, df_test = train_test_split(df, test_size=.2, random_state=42)

    df_train, df_valid = train_test_split(df, test_size=.2, random_state=42)
    
    df_train.to_csv('/Users/gonthierlucas/Desktop/DS_project/IMDB_reviews/data/IMDB_train.csv', index=False)
    df_valid.to_csv('/Users/gonthierlucas/Desktop/DS_project/IMDB_reviews/data/IMDB_valid.csv', index=False)
    df_test.to_csv('/Users/gonthierlucas/Desktop/DS_project/IMDB_reviews/data/IMDB_test.csv', index=False)



if __name__ == "__main__":
    main()