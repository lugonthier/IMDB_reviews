
import pandas as pd
import sys
import os

#To add parent folder access
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from sklearn.pipeline import Pipeline

from preprocessing.text_preprocessor import TextPreprocessor
from preprocessing.vectorizer import Vectorizer
from experiment.experiment import Experiment
from experiment.experiment_type import launch_experiment
from model.base_model import get_model_by_name, get_model_name_by_id

def main():
    if len(sys.argv) < 8:
        usage = "\n Usage: python experimentation/validation.py  mode stopwords  normalization  vectorizer  model_id  new_experiment  experiment_name/experiment_id\
        \n\n\t mode : 1 => simple experiment, 2 => cross validation\
        \
        \n\t stopwords : 0 => No, 1 => 'english' from nltk\
         \n\t normalization : 0 => No, 1 => WordNetLemmatizer, 2 => PorterStemmer \
        \
        \n\t vectorizer : 1 => CountVectorizer, 2 => TfidfVectorizer\
        \n\t model_id : 1 => LogisticRegression, 2 => DecisionTree, 3 => MultinomialNB, 4 => RandomForest, 5 => LinearSVC, 6 => Multi Layer Perceptron, else => All models \
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
    model_id = int(sys.argv[5])
    new_experiment = int(sys.argv[6])

    if(new_experiment == 1):
        experiment_name = str(sys.argv[7])
    else:
        experiment_id = int(sys.argv[7])

    model_names = get_model_name_by_id(model_id)

    all_models = get_model_by_name(model_names)


    
    

    df_train = pd.read_csv('data/IMDB_train.csv')
    

    X_train = df_train.review.to_numpy()
    y_train = df_train.sentiment.apply(lambda x: 0 if x=="negative" else 1).to_numpy()

   

    if new_experiment == 1:
            exp = Experiment(experiment_name=experiment_name) 
    else:
        exp = Experiment(experiment_id=experiment_id)

    text_prep = TextPreprocessor(stopwords=stopwords, normalization=normalization)
    vect = Vectorizer(vectorizer=vectorizer, max_features=32000)

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

        exp.load_data(X_train, y_train,  train, test)
        launch_experiment(exp, mode)
        


if __name__ == "__main__":
    main()