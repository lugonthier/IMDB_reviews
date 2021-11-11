
import numpy as np
import mlflow
from sklearn.model_selection import cross_validate


class Experiment:
    """
    TODO : - handle exception (delete experiment if not processed entirely)
            
    """

    def __init__(self, model=None, experiment_name="",  tracking_uri="http://localhost:5000", experiment_id=None):
        self.name = experiment_name #Must be unique
        self.model = model #whole pipeline to prevent data leakage during cross validation 
        self.tracking_uri = tracking_uri
        
        
        if not experiment_id:
            self.experiment_id = mlflow.create_experiment(name=self.name)
            
        else:
            
            self.experiment_id = experiment_id
            

    

    def run_cross_valid_experimentation(self, X, y, scorers, cv=5, return_train_score=False):
        
        
        mlflow.set_tracking_uri(self.tracking_uri)
        
        with mlflow.start_run(experiment_id=self.experiment_id) as run:
            scores = cross_validate(self.model, X=X, y=y, scoring=scorers, cv=cv, return_train_score=return_train_score)
            
            
            #self.__save_params()
            params = self.model.get_params(deep=True)

            for param, value in params.items():
                mlflow.log_param(param, value)
            
            mlflow.log_param("training size", len(X)/cv)

            for metric, score in scores.items():  
                mlflow.log_metric(metric + '_mean', np.mean(score))
                mlflow.log_metric(metric + '_std', np.std(score))
        
        

        
    def run_simple_experimentation(self, X_train, y_train,  X_test, y_test, prefix="valid", metrics=None):
        """
        TODO : - TEST if cross_val == False then X_test and y_test not None
                - If EXCEPTION delete experiment
        """

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.sklearn.autolog(log_models=False)
        with mlflow.start_run(experiment_id=self.experiment_id):
            
            self.model.fit(X_train, y_train)
            
            #y_pred = self.model.predict(X_test)

            #params = self.model[-1].get_params(deep=False)

            #for param, value in params.items():
            #    mlflow.log_param(param, value)
            mlflow.log_param("training size", len(X_train))
            mlflow.sklearn.eval_and_log_metrics(self.model, X_test, y_test, prefix=prefix)   
            #for key, metric in metrics.items():  
            #    mlflow.log_metric(key, metric(y_pred, y_test))
        

    def __save_params(self):
        pipeline_params = {} 
        for step in self.model:
            pipeline_params[type(step).__name__] = step.get_params(deep=False)

        for step, params in pipeline_params.items():
            for param, value in params.items():
                mlflow.log_param(step+'__'+param, value)

                
        
