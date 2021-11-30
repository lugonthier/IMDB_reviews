from typing import Dict, Tuple


from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier


def get_model_name_by_id(key_model:int) -> Dict:

    models = {1:LogisticRegression.__name__, 2:DecisionTreeClassifier.__name__, 3:MultinomialNB.__name__, 4:RandomForestClassifier.__name__,  5:LinearSVC.__name__, 6:MLPClassifier.__name__}
    if key_model in models.keys():
        
        return models[key_model]
    else:
        return 'ALL'

def get_model_by_name(name:str) -> Dict:
    models = {LogisticRegression.__name__:LogisticRegression(max_iter=500), DecisionTreeClassifier.__name__:DecisionTreeClassifier(max_depth=12),
     MultinomialNB.__name__:MultinomialNB(), RandomForestClassifier.__name__:RandomForestClassifier(max_depth=12),  LinearSVC.__name__:LinearSVC(), MLPClassifier.__name__:MLPClassifier()}

    if name in models.keys():
        single_model = {name: models[name]}
        return single_model
    else:
        return models

def get_params_model_by_name(name: str) -> Dict:

    params = {
            DecisionTreeClassifier.__name__:{'vect__max_features':[4000],
                DecisionTreeClassifier.__name__+'__criterion':['gini', 'entropy'],
                                            DecisionTreeClassifier.__name__+'__splitter':['best', 'random'], DecisionTreeClassifier.__name__+'__max_depth':range(12, 15, 1),
                                            DecisionTreeClassifier.__name__+'__max_features':['auto', 'sqrt', 'log2']},

            MultinomialNB.__name__:{'vect__max_features':[None], MultinomialNB.__name__+'__alpha':[ 0.25, 0.5, 0.75, 1.], MultinomialNB.__name__+'__fit_prior':[True, False]},

            RandomForestClassifier.__name__:{'vect__max_features':[32000], RandomForestClassifier.__name__+'__n_estimators':range(50, 351, 100),
                                             RandomForestClassifier.__name__+'__criterion':['gini', 'entropy'],
                                            RandomForestClassifier.__name__+'__max_depth':range(12, 15, 1), RandomForestClassifier.__name__+'__max_features':['auto', 'sqrt', 'log2']},
                                            
            MLPClassifier.__name__:{'vect__max_features':[400], MLPClassifier.__name__+'__hidden_layer_sizes':[(50,), (50, 100, 50), (50, 100, 200, 100, 50)], MLPClassifier.__name__+'__solver':['lbfgs','adam'], MLPClassifier.__name__+'__alpha':[0.0001, 0.001, 0.01],
            MLPClassifier.__name__+'__learning_rate': ['constant','adaptive'], MLPClassifier.__name__+'__max_iter':[300]},
         
            LinearSVC.__name__:{   'vect__max_features':[4000], LinearSVC.__name__+'__loss':['hinge', 'squared_hinge'],
                                            LinearSVC.__name__+'__C':[.001, .01, 1., 10., 100.],  LinearSVC.__name__+'__max_iter':[1500] },

            LogisticRegression.__name__:{'vect__max_features':[None],
                                        LogisticRegression.__name__+'__C':[  .001, .01, .1, 1., 10., 100.], LogisticRegression.__name__+'__solver':[ 'sag', 'saga'],
                                       LogisticRegression.__name__+'__max_iter':[800]}
        }

    if name in params.keys():
        single_model = {name: params[name]}
        return single_model
    else:
        return params

def get_model_and_params_by_name(name: str) -> Tuple[Dict, Dict]:
    
    return get_model_by_name(name=name), get_params_model_by_name(name=name)

    