import numpy as np
import pandas as pd
import sys
import os

#To add parent folder access
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from sklearn.pipeline import Pipeline

from preprocessing.text_preprocessor import TextPreprocessor
from ensembling.ensembling_model import Ensembling
from preprocessing.vectorizer import Vectorizer
from experiment.experiment import Experiment
from experiment.experiment_type import launch_experiment
from model.base_model import get_model_by_name, get_model_name_by_id

def main():
    if len(sys.argv) < 9:
        usage = "\n Usage: python experimentation/validation.py stopwords  normalization  vectorizer max_features ensembling  model_id  new_experiment  experiment_name/experiment_id\
        \n\t stopwords : 0 => No, 1 => 'english' from nltk\
         \n\t normalization : 0 => No, 1 => WordNetLemmatizer, 2 => PorterStemmer \
        \n\t vectorizer : 1 => CountVectorizer, 2 => TfidfVectorizer\
        \n\t max_features : integer => The maximum features (dimensions) wanted\
        \n\t ensembling : 0 => No, 1 => Yes\
        \n\t model_id : 1 => LogisticRegression, 2 => DecisionTree, 3 => MultinomialNB, 4 => RandomForest, 5 => LinearSVC, 6 => Multi Layer Perceptron, else => All models \
        \n\t new_experiment : 0 => experiment exist, 1: new experiment\
        \n\t experiment_name (if new_experiment==1)\
        \n\t experiment_id (if new_experiment==0)\
        "
        print(usage)
        return

    
    stopwords = [] if int(sys.argv[1]) == 0 else 'english'
    normalization = int(sys.argv[2])
    vectorizer = int(sys.argv[3])
    max_features = int(sys.argv[4])
    ensembling = int(sys.argv[5])
    model_id = int(sys.argv[6])
    new_experiment = int(sys.argv[7])

    if (ensembling == 1):
        ensembling_model = Ensembling()
        all_models = {"Ensemble": ensembling_model}
    else:
        # Get model name(s) by id.
        model_names = get_model_name_by_id(model_id)

        # Get model(s) by name(s).
        all_models = get_model_by_name(model_names)

    if(new_experiment == 1):
        experiment_name = str(sys.argv[8])
    else:
        experiment_id = int(sys.argv[8])

    #Load training set.
    df_train = pd.read_csv('data/IMDB_train.csv')    
    X_train = df_train.review.to_numpy()
    y_train = df_train.sentiment.apply(lambda x: 0 if x=="negative" else 1).to_numpy()

    if new_experiment == 1:
            exp = Experiment(experiment_name=experiment_name) 
    else:
        exp = Experiment(experiment_id=experiment_id)

    text_prep = TextPreprocessor(stopwords=stopwords, normalization=normalization)
    vect = Vectorizer(vectorizer=vectorizer, max_features=max_features)

    #For each model selected. (one model or all)
    for key, model in all_models.items():

        #Create pipeline.
        pipe = Pipeline(steps=[('preprocessor',text_prep), ('vect', vect), ('model', model)])
        exp.model = pipe

        
        train = []
        test = []
        for column in df_train.columns:
            if 'fold' in column:
                train.append(df_train[df_train[column]=='train'].index)
                test.append(df_train[df_train[column]=='test'].index)

        exp.load_data(X_train, y_train,  train, test)
        launch_experiment(exp)
        


if __name__ == "__main__":
    main()