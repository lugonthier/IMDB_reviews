
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, make_scorer



def launch_experiment(exp, mode, X_train, y_train, X_valid=None, y_valid=None, test=False):
   
    if test:
        prefix = 'test_'
    else:
        prefix = 'valid_'
        
    if mode == 1:

            metrics = {accuracy_score.__name__:accuracy_score, f1_score.__name__:f1_score, roc_auc_score.__name__:roc_auc_score}
            exp.run_simple_experimentation(X_train, y_train, X_valid, y_valid, prefix,  metrics)

    elif mode == 2:
        metrics =  {accuracy_score.__name__:make_scorer(accuracy_score), f1_score.__name__:make_scorer(f1_score),
        roc_auc_score.__name__:make_scorer((roc_auc_score))}

        exp.run_cross_valid_experimentation(X_train, y_train, scorers=metrics, return_train_score=True)


def training_size_evaluation(exp, mode, range_step, X_train, y_train, X_valid, y_valid):
    
    for size in range(range_step, len(X_train), range_step):
        X_train_sized = X_train[:size]
        y_train_sized = y_train[:size]

        launch_experiment(exp, mode, X_train_sized, y_train_sized, X_valid, y_valid)


def dimensionality_size_evaluation(exp,  mode, range_step, X_train, y_train, X_valid, y_valid, max_dim = 102000):#After investigation, maximum possible dimension is 101895
        
    for dim in range(range_step, max_dim, range_step):

        exp.model[1].max_features = dim
        launch_experiment(exp, mode, X_train, y_train, X_valid, y_valid)