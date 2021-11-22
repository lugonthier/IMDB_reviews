import pandas as pd
import numpy as np
import mlflow

from typing import Dict, List, Union

from time import time

class Experiment:
    """Class to track machine learning experiment. Based on the open source project mlflow.

    Args:
        name : str
            Is the experiment name.

        model : sklearn.pipeline.Pipeline
            The whole pipeline from preprocessing to machine learning model.

        tracking_uri : str
            The uri used to store experiment results.
        
        experiment_id : int
            The (unique) experiment id.
    """

    def __init__(self, model=None, experiment_name:str = "",  tracking_uri:str = "http://localhost:5000", experiment_id: int = None):
        self.name = experiment_name #Must be unique
        self.model = model #whole pipeline to prevent data leakage during cross validation 
        self.tracking_uri = tracking_uri
        
        
        if not experiment_id:
            self.experiment_id = mlflow.create_experiment(name=self.name)
            
        else:
            self.experiment_id = experiment_id
            
    def load_data(self, X_train:Union[List, np.ndarray, pd.Series], y_train:Union[List, np.ndarray, pd.Series],  train_indexes:Union[List, np.ndarray, pd.Series]=None,
             test_indexes:Union[List, np.ndarray, pd.Series]=None, X_test:Union[List, np.ndarray, pd.Series]=None,
            y_test:Union[List, np.ndarray, pd.Series]=None) -> None:
        """ This method load all data that will be used. Note that either (X_test,y_test) or (train_indexes,test_indexes) should be set.

        Args:
            X_train (Union[List, np.ndarray, pd.Series]): Data used to train the model.
            y_train (Union[List, np.ndarray, pd.Series]): Target corresponding to X_train.
            X_test (Union[List, np.ndarray, pd.Series]): Data used to evaluate the model. Defaults to None.
            y_test (Union[List, np.ndarray, pd.Series]): Target corresponding to X_test. Defaults to None.
            train_indexes (Union[List, np.ndarray, pd.Series], optional): Indices of train folders in X_train for cross validation. Defaults to None.
            test_indexes (Union[List, np.ndarray, pd.Series], optional): Indices of test folder in X_test for cross validation. Defaults to None.
        """

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.train_indexes = train_indexes
        self.test_indexes = test_indexes

    
    def run_cross_valid_experimentation(self, metrics:Dict, prefix:str) -> None:
        """This method is used to run a cross validation. Model parameters and model scores are saved.

        Args:
            metrics (Dict): metrics used to evaluate the model.
            prefix (str): To decide if it's test or validation purpose.
        """
        
        mlflow.set_tracking_uri(self.tracking_uri)
        
        with mlflow.start_run(experiment_id=self.experiment_id) as run:
            
            scores = {}
            for metric_name in metrics.keys():
                scores['train_'+metric_name] = []
                scores[prefix+metric_name] = []


            fit_time = []
            predict_time = []
            for fold_index in range(len(self.train_indexes)):

                training_time0 = time()
                unique, counts = np.unique(self.y_train[self.train_indexes[fold_index]], return_counts=True)
                print(dict(zip(unique, counts)))
                print('-'*20)
                self.model.fit(self.X_train[self.train_indexes[fold_index]], self.y_train[self.train_indexes[fold_index]])
                training_time1 = time()
                y_pred_train = self.model.predict(self.X_train[self.train_indexes[fold_index]])

                testing_time0 = time()
                y_pred_test = self.model.predict(self.X_train[self.test_indexes[fold_index]])
                testing_time1 = time()

                fit_time.append(training_time1 - training_time0)
                predict_time.append(testing_time1 - testing_time0)

                for metric_name, metric in metrics.items():
                    scores['train_'+metric_name].append(metric(y_pred_train, self.y_train[self.train_indexes[fold_index]]))
                    scores[prefix+metric_name].append(metric(y_pred_test, self.y_train[self.test_indexes[fold_index]]))
            
            scores['fit_time'] = fit_time
            scores['predict_time'] = predict_time
            #self.__save_params()
            params = self.model.get_params(deep=True)

            for param, value in params.items():
                mlflow.log_param(param, value)
            
            mlflow.log_param("training_size", len(self.train_indexes[fold_index]))
            
            for metric, score in scores.items():  
                mlflow.log_metric(metric + '_mean', np.mean(score))
                mlflow.log_metric(metric + '_std', np.std(score))
        
            
        
    def run_simple_experimentation(self, prefix:str="valid_", metrics:Dict=None) -> None:
        """This method is used to run a simple validation. Model parameters and model scores are saved.

        Args:
            prefix (str, optional): to prefix metrics name. Defaults to "valid".
            metrics (Dict, optional): sklearn metrics. Defaults to None.
        """

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.sklearn.autolog(log_models=False)
        with mlflow.start_run(experiment_id=self.experiment_id):
            
            self.model.fit(self.X_train, self.y_train)
            
            
            mlflow.log_param("training size", len(self.X_train))
            mlflow.sklearn.eval_and_log_metrics(self.model, self.X_test, self.y_test, prefix=prefix)   
            #for key, metric in metrics.items():  
            #    mlflow.log_metric(key, metric(y_pred, y_test))
        

    def __save_params(self):
        pipeline_params = {} 
        for step in self.model:
            pipeline_params[type(step).__name__] = step.get_params(deep=False)

        for step, params in pipeline_params.items():
            for param, value in params.items():
                mlflow.log_param(step+'__'+param, value)

                
        
