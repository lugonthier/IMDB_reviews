import pandas as pd
import sys
import os

#To add parent folder access
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.pipeline import Pipeline

from preprocessing.text_preprocessor import TextPreprocessor
from preprocessing.vectorizer import select_vectorizer
from experiment.experiment import Experiment
from experiment.experiment_type import launch_experiment

def main():
    if len(sys.argv) < 7:
        usage = "\n Usage: python main.py  mode  model_selected  new_experiment  experiment_name/experiment_id  evaluation\
        \n\n\t mode : 1 => simple experiment, 2 => cross validation\
        \
        \n\t stopwords : 0 => No, 1 => 'english' from nltk\
         \n\t normalization : 0 => No, 1 => WordNetLemmatizer, 2 => PorterStemmer \
        \
        \n\t vectorizer : 1 => CountVectorizer, 2 => TfidfVectorizer\
        \
        \n\t new_experiment : 0 => experiment exist, 1: new experiment\
        \n\t experiment_name (if new_experiment==1)\
        \n\t experiment_id (if new_experiment==0)\
        "
        print(usage)
        return

    mode = int(sys.argv[1])
    stopwords = [] if int(sys.argv[2]) == 0 else 'english'
    normalization = int(sys.argv[3])

    vectorizer = int(sys.argv[4])
    new_experiment = int(sys.argv[5])

    if(new_experiment == 1):
        experiment_name = str(sys.argv[6])
    else:
        experiment_id = int(sys.argv[6])

    all_models = {1:LogisticRegression(), 2:DecisionTreeClassifier(), 3:MultinomialNB(),  5:LinearSVC(), 6:MLPClassifier()}
    

    df_train = pd.read_csv('data/IMDB_train.csv')
    df_valid = pd.read_csv('data/IMDB_valid.csv')

    X_train = df_train.review.to_numpy()
    y_train = df_train.sentiment.apply(lambda x: 0 if x=="negative" else 1).to_numpy()

    X_valid = df_valid.review.to_numpy()
    y_valid = df_valid.sentiment.apply(lambda x: 0 if x=="negative" else 1).to_numpy()

    

    if new_experiment == 1:
            exp = Experiment(experiment_name=experiment_name) 
    else:
        exp = Experiment(experiment_id=experiment_id)

    text_prep = TextPreprocessor(stopwords=stopwords, normalization=normalization)
    vect = select_vectorizer(vectorizer)

    for key, model in all_models.items():
        pipe = Pipeline(steps=[('preprocessor',text_prep), ('vect', vect), ('model', model)])
        exp.model = pipe
        
        if mode == 2:
            train = []
            test = []
            for column in df_train.columns:
                if 'fold' in column:
                    train.append(df_train[df_train[column]=='train'].index)
                    test.append(df_train[df_train[column]=='test'].index)

        launch_experiment(exp, mode, X_train, y_train, X_valid, y_valid, train_indexes=train, test_indexes=test)
        


if __name__ == "__main__":
    main()