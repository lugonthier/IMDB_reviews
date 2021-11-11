import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, make_scorer
from experiment import Experiment






def main():
    if len(sys.argv) < 5:
        usage = "\n Usage: python classifieur.py method nb_train nb_test lambda bruit corruption don_ab\
        \n\n\t mode : 1 => simple experiment, 2 => cross validation\
        \n\t model_seleted : 1 => Logistic Regression, 2 => ... until 6, 7 => Performe all model ?\
        \n\t new_experiment : 0 => experiment exist, 1: new experiment\
        \n\t experiment_name (if new_experiment==1)\
        \n\t experiment_id (if new_experiment==0)\
        \n\t range_step : step to iterate over dataset size"
        print(usage)
        return

    mode = int(sys.argv[1])
    model_selected = int(sys.argv[2])
    new_experiment = int(sys.argv[3]) # 0:no, 1:yes

    if(new_experiment == 1):
        experiment_name = str(sys.argv[4])
    else:
        experiment_id = int(sys.argv[4])

    range_step = int(sys.argv[5])
    

    models = {1:LogisticRegression()}
    if (model_selected > 0) and (model_selected < 7):
        model = models[model_selected]
    
    df = pd.read_csv('/Users/gonthierlucas/Desktop/DS_project/IMDB_reviews/data/IMDB_Dataset.csv')

    if new_experiment == 1:
            exp = Experiment( experiment_name=experiment_name)
            
    else:
        exp = Experiment( experiment_id=experiment_id)

    for size in range(range_step, len(df), range_step):

        X = df.review.to_numpy()[:size]
        y = df.sentiment.apply(lambda x: 0 if (x == 'negative') else 1).to_numpy()[:size]
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        cv = CountVectorizer()

        pipe = Pipeline(steps=[('vect', cv), ('model', model)])

        exp.model = pipe
        

        if mode == 1:
            metrics = {accuracy_score.__name__:accuracy_score, f1_score.__name__:f1_score, roc_auc_score.__name__:roc_auc_score}
            exp.run_simple_experimentation(X_train, y_train, X_test, y_test, "test_",  metrics)

        elif mode == 2:
            metrics =  {accuracy_score.__name__:make_scorer(accuracy_score), f1_score.__name__:make_scorer(f1_score),
            roc_auc_score.__name__:make_scorer((roc_auc_score))}

            exp.run_cross_valid_experimentation(X_train, y_train, scorers=metrics, return_train_score=True)


    
if __name__=="__main__":
    main()