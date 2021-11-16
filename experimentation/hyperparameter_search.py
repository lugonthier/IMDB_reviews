import preprocessing
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from preprocessing.text_preprocessor import TextPreprocessor
from preprocessing.vectorizer import select_vectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression


def main():
    params = {DecisionTreeClassifier.__name__:DecisionTreeClassifier(), LinearSVC.__name__:LinearSVC(),
          
           LogisticRegression.__name__:LogisticRegression()
         }

    models = {}

    df_train = pd.read_csv('/Users/gonthierlucas/Desktop/DS_project/IMDB_reviews/data/train_Dataset.csv')
    X = df_train.review.to_numpy()
    y = df_train.sentiment.apply(lambda x: 0 if x=="negative" else 1).to_numpy()

    #We do text preprocessing before the search of hyperparameters 
    #Because data transformed will not be dependent of data fitted
    #Contrary to the vectorizers.

    preprocessor = TextPreprocessor()
    X_transformed = preprocessor.transform(X)

    for key, model in models.items():

        pipe = Pipeline(steps=[('vectorizer', select_vectorizer(1)), (key, model)])


        grid = GridSearchCV()
        grid.fit(X_transformed, y)



if __name__== "__main__":
    main()