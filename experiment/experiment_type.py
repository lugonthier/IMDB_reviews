
import numpy as np
from experiment.experiment import Experiment
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score



def launch_experiment(exp:Experiment, mode:int, test:bool=False) -> None:
    """This function help launching experiment (validation or cross validation).

    Args:
        exp (Experiment): An Experiment object already initialized with his pipeline and data.
        mode (int): 1 => simple validation, 2 => cross validation.
        test (bool, optional): To specified if it's the final test or a validation. Must be used only for simple experiment. Defaults to False.
    """
    if test:
        prefix = 'test_'
    else:
        prefix = 'valid_'
    metrics = {accuracy_score.__name__:accuracy_score, f1_score.__name__:f1_score, roc_auc_score.__name__:roc_auc_score}
    if mode == 1:

           
            exp.run_simple_experimentation( prefix,  metrics)

    elif mode == 2:

        exp.run_cross_valid_experimentation(metrics, prefix)


def training_size_evaluation(exp:Experiment, mode:int, range_step:int) -> None:
    """Function that evaluate models according to the number of training sample.
        Because k folds are already made based on the real train set size, this function take a
        subset of the train set (at each iteration until we use the maximum size). 

    Args:
        exp (Experiment): An Experiment object already initialized with his pipeline and data.
        mode (int): 1 => simple validation, 2 => cross validation.
        range_step (int): step to increase number of training example.
    """
    
    
    y_train = exp.y_train
    train_indexes = exp.train_indexes

    #We get the target values of each train k-fold. 
    y_train_kfold = [y_train[fold_train] for fold_train in train_indexes]
    
    for size in range(range_step, len(train_indexes[0]) + range_step, range_step):

        if size > len(train_indexes[0]):
            #Last iteration, with the maximum train set size.
            exp.train_indexes = train_indexes

        else:
            #In order to conserve a balance data set.
            k0 = int(size/2)
            k1 = int(size/2)
        
            #Select randomly the indexes indices. k0 of class 0 and k1 of class 1. And this for each k-fold.
            indices_of_indexes = np.array([np.concatenate( (np.random.choice(np.where(y == 0)[0], k0, replace=False), np.random.choice(np.where(y == 1)[0], k1, replace=False))) for y in y_train_kfold])
        

            #Then we select the indexes based on the previous indices generated.
            exp.train_indexes = np.array( [ train_indexes[index][indices_of_indexes[index]] for index in range(len(train_indexes))],dtype='int32')


        launch_experiment(exp, mode)

    
 

def dimensionality_size_evaluation(exp:Experiment,  mode:int, range_step:int, max_dim:int = 102000, linear=False) -> None:#After investigation, maximum possible dimension is 101895
    """Function that evaluate models according to the number of dimensionality. 
    Args:
        exp (Experiment): An Experiment object already initialized with his pipeline and data.
        mode (int): 1 => simple validation, 2 => cross validation.
        range_step (int): step to increase number of dimension.
        max_dim (int, optional): maximum dimension wanted (or possible). Defaults to 102000.
    """
    if linear:
        for dim in range(range_step, max_dim, range_step):

            exp.model[1].update_params(max_features=dim)
            launch_experiment(exp, mode)
    else:
        dim = range_step
        while dim < max_dim:
            exp.model[1].update_params(max_features=dim)
            launch_experiment(exp, mode)

            dim = 2*dim

