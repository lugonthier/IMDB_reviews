
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, make_scorer



def launch_experiment(exp, mode, test=False):
   
    if test:
        prefix = 'test_'
    else:
        prefix = 'valid_'
    metrics = {accuracy_score.__name__:accuracy_score, f1_score.__name__:f1_score, roc_auc_score.__name__:roc_auc_score}
    if mode == 1:

            metrics = {accuracy_score.__name__:accuracy_score, f1_score.__name__:f1_score, roc_auc_score.__name__:roc_auc_score}
            exp.run_simple_experimentation( prefix,  metrics)

    elif mode == 2:
        #metrics =  {accuracy_score.__name__:make_scorer(accuracy_score), f1_score.__name__:make_scorer(f1_score),
        #roc_auc_score.__name__:make_scorer((roc_auc_score))}

        exp.run_cross_valid_experimentation(metrics, prefix)


def training_size_evaluation(exp, mode, range_step):
    #This function doesn't work with cross validation for now.
    X_train = exp.X_train
    y_train = exp.y_train
    

    for size in range(range_step, len(X_train), range_step):
        exp.X_train = X_train[:size]
        exp.y_train = y_train[:size]
       

        launch_experiment(exp, mode)

    exp.X_train = X_train
    exp.y_train = y_train
 

def dimensionality_size_evaluation(exp,  mode, range_step, max_dim = 102000):#After investigation, maximum possible dimension is 101895
        
    for dim in range(range_step, max_dim, range_step):

        exp.model[1].max_features = dim
        launch_experiment(exp, mode)

