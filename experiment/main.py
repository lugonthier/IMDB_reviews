
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
from experiment.experiment_type import training_size_evaluation, dimensionality_size_evaluation, launch_experiment




def main():
    if len(sys.argv) < 9:
        usage = "\n Usage: python main.py  mode  model_selected  new_experiment  experiment_name/experiment_id  evaluation\
        \n\n\t mode : 1 => simple experiment, 2 => cross validation\
        \
        \n\t stopwords : 0 => No, 1 => 'english' from nltk\
         \n\t normalization : 0 => No, 1 => WordNetLemmatizer, 2 => PorterStemmer \
        \
        \n\t vectorizer : 1 => CountVectorizer, 2 => TfidfVectorizer\
        \n\t max_features : integer\
        \
        \n\t model_seleted : 1 => Logistic Regression, 2 => ... until 6, 7 => Performe all model ?\
        \
        \n\t new_experiment : 0 => experiment exist, 1: new experiment\
        \n\t experiment_name (if new_experiment==1)\
        \n\t experiment_id (if new_experiment==0)\
        \n\t evaluation: 0 => normal, 1 => training size, 2 => dimensionality\
        "
        print(usage)
        return

    mode = int(sys.argv[1])
    stopwords = [] if int(sys.argv[2]) == 0 else 'english'
    normalization = int(sys.argv[3])

    vectorizer = int(sys.argv[4])
    max_features = int(sys.argv[5])

    model_selected = int(sys.argv[6])

    new_experiment = int(sys.argv[7]) # 0:no, 1:yes

    if(new_experiment == 1):
        experiment_name = str(sys.argv[8])
    else:
        experiment_id = int(sys.argv[8])

    evaluation = int(sys.argv[9])
    
    if evaluation in [1, 2]: #Ask for training step
        isint = False
        while not isint:
            try:
                range_step = int(input("Please, insert the step : \n\n"))
                isint = True
            except:
                print("Please give an int as step.")
            
    
        

    all_models = {1:LogisticRegression(), 2:DecisionTreeClassifier(), 3:MultinomialNB(),  5:LinearSVC(), 6:MLPClassifier()}
    models = []
    if (model_selected > 0) and (model_selected < 7):
        models.append(all_models[model_selected])
    else:
        for key, model in all_models.items():
            models.append(model)




    df = pd.read_csv('data/IMDB_Dataset.csv')

    if new_experiment == 1:
            exp = Experiment(experiment_name=experiment_name) 
    else:
        exp = Experiment(experiment_id=experiment_id)


    text_prep = TextPreprocessor(stopwords=stopwords, normalization=normalization)
    vect = select_vectorizer(vectorizer)

    for model in models:
        pipe = Pipeline(steps=[('preprocessor',text_prep), ('vect', vect), ('model', model)])
        exp.model = pipe

        if evaluation == 1:
            training_size_evaluation(exp, df, mode, range_step)

        elif evaluation == 2:
            dimensionality_size_evaluation(exp, df, mode, range_step)

        else:
            X = df.review.to_numpy()
            y = df.sentiment.apply(lambda x: 0 if (x == 'negative') else 1).to_numpy()
            launch_experiment(exp, mode, X, y)

    
if __name__=="__main__":
    main()