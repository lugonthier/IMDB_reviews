import sys
import os

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import pandas as pd
from model.base_model import get_model_name_by_id, get_model_by_name
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score
from preprocessing.text_preprocessor import TextPreprocessor
from preprocessing.vectorizer import Vectorizer


def main():
    train = pd.read_csv('data/IMDB_train.csv')
    test = pd.read_csv('data/IMDB_test.csv')

    all_names = []
    for i in range(1, 7):
        all_names.append(get_model_name_by_id(i))

    all_models = []
    for name in all_names:
        all_models.append(get_model_by_name(name)[name])

    #best dimensions found for each model
    max_features = [4000, 32000, None, 32000, 8000, 2000]

    x_train = train.review.to_numpy()
    y_train = train.sentiment.apply(lambda sent: 0 if sent == 'negative' else 1).to_numpy()

    x_test = test.review.to_numpy()
    y_test = test.sentiment.apply(lambda sent: 0 if sent == 'negative' else 1).to_numpy()

    list_accuracy = []
    tn_list = []
    fp_list = []
    fn_list = []
    tp_list = []
    for i in range(len(all_models)):
        text_prep = TextPreprocessor(stopwords='english', normalization=2)  # remove stop words + PorterStemmer
        vectorizer = Vectorizer(2, max_features=max_features[i]) #TF-IDF Vectorizer

        pipe = Pipeline([('preprocessor', text_prep), ('vect', vectorizer), (all_names[i], all_models[i])])
        pipe.fit(x_train, y_train)

        y_pred = pipe.predict(x_test)

        print(all_names[i])
        print('accuracy score : ', accuracy_score(y_test, y_pred))
        print('confusion matrix', confusion_matrix(y_test, y_pred))
        print('-' * 20)

        list_accuracy.append(accuracy_score(y_test, y_pred))
        tn_list.append(confusion_matrix(y_test, y_pred).ravel()[0])
        fp_list.append(confusion_matrix(y_test, y_pred).ravel()[1])
        fn_list.append(confusion_matrix(y_test, y_pred).ravel()[2])
        tp_list.append(confusion_matrix(y_test, y_pred).ravel()[3])

    dict_res = {'model': all_names, 'accuracy': list_accuracy, 'TN': tn_list, 'FP': fp_list, 'FN': fn_list, 'TP': tp_list}
    results = pd.DataFrame(dict_res)
    # noinspection PyTypeChecker
    results.to_csv('result_analysis/results/final_test/data_results/testing_results.csv')


if __name__ == '__main__':
    main()