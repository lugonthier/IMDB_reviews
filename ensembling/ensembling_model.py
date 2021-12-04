import sys
import numpy as np

current_module = sys.modules[__name__]
from sklearn.base import BaseEstimator
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from model.base_model import get_model_by_name, get_model_name_by_id


class Ensembling(BaseEstimator):
    """Class used to perform model ensembling.
    Args:
        BaseEstimator ([type]): inherit from scikit-learn BaseEstimator to fit in pipeline properly.
    """

    def __init__(self):
        """ init Ensembling model.

        Args:
            ensembling_model : int
                if 1 => Adaboost is used.
                if 2 => VotingClassifier is used.

            all_models : List | array
                the list of models on which ensembling will be performed on
        """
        self.ensembling_model, self.all_models = self.choose_ensembling_model()

        if self.ensembling_model == 1:
            self.ensembling_model = AdaBoostClassifier(list(self.all_models[0].values())[0], n_estimators=100,
                                                       random_state=0)
        elif self.ensembling_model == 2:
            estimators = []
            for model_dict in self.all_models:
                for key, model in model_dict.items():
                    estimators.append((key, model))
            self.ensembling_model = VotingClassifier(estimators=estimators, voting='soft')

    def fit(self, X, y=None):
        """fit method.

        Args:
            X ([type]): The training data.
            y ([type], optional): The training targets. Defaults to None.

        Returns:
            [type]: The bottom level vectorizer (Count or Tfidf from scikit-learn).
        """

        return self.ensembling_model.fit(X, y)

    def predict(self, X, y=None):
        """Transform method.

        Args:
            X ([type]): The data to be transformed.
            y ([type], optional): Targets. Defaults to None.

        Returns:
            [type]: The data transformed.
        """
        return self.ensembling_model.predict(X)

    def choose_ensembling_model(self):
        ensembling_array = []
        model_names_array = []
        all_models_array = []

        print("You choose model ensembling, select an ensembling method : ")
        print("1 =>Adaboost, 2 => VotingClassifier")

        ensembling_model = int(input())

        print("Please choose models to combine : ")
        print("0 => stop, 1 => LogisticRegression, 2 => DecisionTree, 3 => MultinomialNB, 4 => RandomForest,"
              " 5 => ""LinearSVC, 6 => Multi Layer Perceptron")
        while True:
            try:
                model_id_input = int(input())
                if model_id_input == 0:
                    if len(ensembling_array) == 0:
                        print("Select at least one model : ")
                    else:
                        ensembling_array = np.array(ensembling_array)
                        break
                else:
                    """if (model_id_input in ensembling_array):
                        print("Model is already choosen")"""
                    if ensembling_model == 1:
                        ensembling_array.append(model_id_input)
                        print("Model successfully added")
                        break
                    else:
                        ensembling_array.append(model_id_input)
                        print("Model successfully added")
            except:
                print("That's not a valid option!")

        for val in ensembling_array:
            model_names_array.append(get_model_name_by_id(val))
        for model_name in model_names_array:
            all_models_array.append(get_model_by_name(model_name))

        return ensembling_model, all_models_array
