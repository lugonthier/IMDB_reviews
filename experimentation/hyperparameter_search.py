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
from preprocessing.vectorizer import Vectorizer
from model.base_model import get_model_and_params_by_name, get_model_name_by_id



def main():
    if len(sys.argv) < 2:
        usage = "\n Usage: python hyperparameter_search.py model_selected\
            \n\n\t vectorizer : 1 => CountVectorizer, 2 => TfidfVectorizer\
        \n\n\t model_selected : 1 => LogisticRegression, 2 => DecisionTree, 3 => MultinomialNB, 4 => RandomForest, 5 => LinearSVC, 6 => Multi Layer Perceptron, else => All models \
        "
        print(usage)
        return

    stopwords = [] if int(sys.argv[1]) == 0 else 'english'
    normalization = int(sys.argv[2])
    vectorizer = int(sys.argv[3])
    model_selected = int(sys.argv[4])

    
    model_names = get_model_name_by_id(model_selected)
    
    models, params = get_model_and_params_by_name(model_names)


    df_train = pd.read_csv('data/IMDB_train.csv')
    X = df_train.review.to_numpy()
    y = df_train.sentiment.apply(lambda x: 0 if x=="negative" else 1).to_numpy()

    #We do text preprocessing before the search of hyperparameters 
    #Because data transformed will not be dependent of data fitted
    #Contrary to the vectorizers.

    preprocessor = TextPreprocessor(stopwords=stopwords, normalization=normalization)
    X_transformed = preprocessor.transform(X)

    
    #Handle several model tuning but should be just one model unless there are few parameters.
    for key, model in models.items():
        print(key, model, params)
        pipe = Pipeline(steps=[('vectorizer', Vectorizer(vectorizer)), (key, model)])


        grid = GridSearchCV(pipe, params[key], verbose=1)

        grid.fit(X_transformed, y)

        results = pd.DataFrame(grid.cv_results_)

        results.to_csv('result_analysis/results/hyperparameter_search/'+ key + '.csv')

if __name__== "__main__":
    main()