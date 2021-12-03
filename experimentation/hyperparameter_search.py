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
    """
    This script is used to perform hyperparameter tuning.
    See below to choose model and preprocessing configuration.
    """

    if len(sys.argv) < 5:
        usage = "\n Usage: python hyperparameter_search.py stopwords normalization vectorizer model_selected\
            \n\n\t stopwords : 0 => No, 1 => 'english' from nltk\
            \n\n\t normalization : 0 => No, 1 => WordNetLemmatizer, 2 => PorterStemmer \
            \n\n\t vectorizer : 1 => CountVectorizer, 2 => TfidfVectorizer\
        \n\n\t model_selected : 1 => LogisticRegression, 2 => DecisionTree, 3 => MultinomialNB, 4 => RandomForest, 5 => LinearSVC, 6 => Multi Layer Perceptron, else => All models \
        "
        print(usage)
        return

    stopwords = [] if int(sys.argv[1]) == 0 else 'english'
    normalization = int(sys.argv[2])
    vectorizer = int(sys.argv[3])
    model_selected = int(sys.argv[4])

    #Get model name based on integer selected.
    model_names = get_model_name_by_id(model_selected)
    
    #Get model and hyperparameters to test.
    models, params = get_model_and_params_by_name(model_names)

    #Load training data
    df_train = pd.read_csv('data/IMDB_train.csv')
    X = df_train.review.to_numpy()
    y = df_train.sentiment.apply(lambda x: 0 if x=="negative" else 1).to_numpy()

    
    preprocessor = TextPreprocessor(stopwords=stopwords, normalization=normalization)
    
    #Handle several model tuning but should be just one model unless there are very few parameters.
    for key, model in models.items():
        print(key, model, params)
        
        pipe = Pipeline(steps=[('preprocessor', preprocessor),('vect', Vectorizer(vectorizer=vectorizer).vectorizer), (key, model)])


        grid = GridSearchCV(pipe, params[key], verbose=2, n_jobs=3)

        grid.fit(X, y)

        results = pd.DataFrame(grid.cv_results_)

        #The results of the hyperparameter search (tuning) are save.
        results.to_csv('result_analysis/results/hyperparameter_search/data_results/'+ key + '.csv')

if __name__== "__main__":
    main()