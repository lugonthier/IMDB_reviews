
import pandas as pd
import numpy as np

import sys
import os


#To add parent folder access
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from preprocessing.text_preprocessor import TextPreprocessor
from preprocessing.vectorizer import select_vectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression


def main():
    if len(sys.argv) < 2:
        usage = "\n Usage: python hyperparameter_search.py model_selected\
        \n\n\t model_selected : 1 => LogisticRegression, 2 => LinearSVC, ..\
        "
        print(usage)
        return

    model_selected = int(sys.argv[1])

    if model_selected == 1:
        model_selected = LogisticRegression.__name__
    elif model_selected == 2:
        model_selected = DecisionTreeClassifier.__name__
    elif model_selected == 3:
        model_selected = LinearSVC.__name__


    models = {DecisionTreeClassifier.__name__:DecisionTreeClassifier(), LinearSVC.__name__:LinearSVC(),
          
           LogisticRegression.__name__:LogisticRegression()
        }

    params = {
            DecisionTreeClassifier.__name__:{ DecisionTreeClassifier.__name__+'__criterion':['gini', 'entropy'],
                                            DecisionTreeClassifier.__name__+'__splitter':['best', 'random'], DecisionTreeClassifier.__name__+'__max_depth':range(2, 14, 1),
                                            DecisionTreeClassifier.__name__+'__max_features':['auto', 'sqrt', 'log2']},
         
            LinearSVC.__name__:{   LinearSVC.__name__+'__loss':['hinge', 'squared_hinge'], LinearSVC.__name__+'__tol':[.000001, .00001, .0001, .001],
                                            LinearSVC.__name__+'__C':[.001, .01, 1., 10., 100.], LinearSVC.__name__+'__max_iter':range(800, 2100, 100) },
            LogisticRegression.__name__:{  LogisticRegression.__name__+'__tol':[.000001, .00001], LogisticRegression.__name__+'__C':[.001, .01], LogisticRegression.__name__+'__solver':[ 'sag', 'saga'],
                                        LogisticRegression.__name__+'__max_iter':range(150, 851, 300)},
         }


    df_train = pd.read_csv('data/IMDB_train.csv')
    X = df_train.review.to_numpy()
    y = df_train.sentiment.apply(lambda x: 0 if x=="negative" else 1).to_numpy()

    #We do text preprocessing before the search of hyperparameters 
    #Because data transformed will not be dependent of data fitted
    #Contrary to the vectorizers.

    preprocessor = TextPreprocessor()
    X_transformed = preprocessor.transform(X)

    

    pipe = Pipeline(steps=[('vectorizer', select_vectorizer(1)), (model_selected, models[model_selected])])


    grid = GridSearchCV(pipe, params[model_selected], verbose=2)

    grid.fit(X_transformed, y)

    results = pd.DataFrame(grid.cv_results_)

    results.to_csv( 'data/results/hyperparameter_search/'+ model_selected + '.csv')

if __name__== "__main__":
    main()