import pandas as pd
import sys
import os

#To add parent folder access
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

from preprocessing.text_preprocessor import TextPreprocessor
from experiment import Experiment
from experiment_type import training_size_evaluation




def main():
    if len(sys.argv) < 5:
        usage = "\n Usage: python main.py  mode  model_selected  new_experiment  experiment_name/experiment_id  evaluation\
        \n\n\t mode : 1 => simple experiment, 2 => cross validation\
        \n\t model_seleted : 1 => Logistic Regression, 2 => ... until 6, 7 => Performe all model ?\
        \n\t new_experiment : 0 => experiment exist, 1: new experiment\
        \n\t experiment_name (if new_experiment==1)\
        \n\t experiment_id (if new_experiment==0)\
        \n\t evaluation: 0=> normal, 1 => training size, 2 => dimensionality"
        print(usage)
        return

    mode = int(sys.argv[1])
    model_selected = int(sys.argv[2])
    new_experiment = int(sys.argv[3]) # 0:no, 1:yes

    if(new_experiment == 1):
        experiment_name = str(sys.argv[4])
    else:
        experiment_id = int(sys.argv[4])

    evaluation = int(sys.argv[5])
    
    if evaluation == 1: #Ask for training step
        isint = False
        while not isint:
            try:
                range_step = int(input("Please, insert the step : \n\n"))
                isint = True
            except:
                print("Please give an int as step.")
            
    elif evaluation == 2:
        print("evaluate dimensionality")
        #TODO
        

    models = {1:LogisticRegression()}
    if (model_selected > 0) and (model_selected < 7):
        model = models[model_selected]
    




    df = pd.read_csv('/Users/gonthierlucas/Desktop/DS_project/IMDB_reviews/data/IMDB_Dataset.csv')

    if new_experiment == 1:
            exp = Experiment(experiment_name=experiment_name) 
    else:
        exp = Experiment(experiment_id=experiment_id)


    text_prep = TextPreprocessor(normalization=1)
    cv = CountVectorizer()

    pipe = Pipeline(steps=[('preprocessor',text_prep), ('vect', cv), ('model', model)])
    exp.model = pipe

    training_size_evaluation(exp, df, mode, range_step)

    
if __name__=="__main__":
    main()