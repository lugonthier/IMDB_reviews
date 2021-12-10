# IMDB_reviews

Sentiment analysis of 50K movie reviews from [IMDb](https://www.imdb.com).

This project was done as a part of the [IFT712 - Technique d'apprentissage](https://www.usherbrooke.ca/admission/fiches-cours/IFT712?fp=005) at Sherbrooke University.

projet members :
 - Lucas Gonthier
    - lugonthier (github account)
    - gonl3002 (gitlab account)
- Adrien Ingarao
    - inga2603 (gitlab account)

# Features

the tracking of model results is based on the open source project [mlflow](https://mlflow.org).

Note that for the **cross validation** and **size evaluation** features, you can found the results :
 - launch the command line ``mlflow ui`` in the project directory (IMDB_reviews) in a terminal.
 - then go to ``http://localhost:5000`` in a browser, so make sure to not have something running on port 5000.
 - Then you can launch the following scripts below in an other terminal, you will see the results in real time in the ``http://localhost:5000`` URI.
 - If you can't do it, please refer to the MLflow documentation at the [**Viewing the Tracking UI**](https://www.mlflow.org/docs/latest/quickstart.html) section.


## Cross validation
This script can be found to [validation.py](experimentation/validation.py).
To perform a (stratified) cross validation. 

```python experimentation/validation.py {stopwords} {normalization} {vectorizer} {max_features} {ensembling} {model_id} {new_experiment} {experiment}```

For example, ``python experimentation/validation.py 1 2 2 3000 0 1 1 LogisticRegressionTest``

where :
- stopwords : 
    - **0** :  To not remove the stop words.
    - "**english**" : To remove english stop words. nltk stop words will be used.

- normalization :
    - **0** : To not perform normalization.
    - **1** : To perform lemmatizing. The **WordNetLemmatizer** from nltk will be used.
    - **2** : Top perform stemming. The **PorterStemmer** from nltk will be used.

- vectorizer :
    - **1** : To use a count vectorizer.
    - **2** : To use a tfidf vectorizer.

- max_features :
    - an integer to decide the maximum features (dimensionality of the data).

- ensembling :
    - **0** To use classic models (no ensemble).
    - **1** To use the ensembling models.

- model_id :
    - **1** : To use Logistic Regresstion.
    - **2** : To use Decision Tree.
    - **3** : To use Multinomial NB.
    - **4** : To use Random Forest.
    - **5** : To use Linear SVC.
    - **6** : To use Multi Layer Perceptron.
    - **7** : To use **all** model.

- new_experiment :
    - **0** : To use an existing experiment.
    - **1** : To create a new experiment.

- experiment :

    - **ID** (integer) of the experiment (if new_experiment = 0).
    - **name** (string) to name the new experiment (if new_experiment = 1).

Note that in order to compare all model on the exact same data :
-  You can run all model at once (model_id = 7)
- Or you can execute the command for each model but make sure to not modified or re-split the data ([split_data.py](experimentation/split_data.py)).


## Fine tuning

To perform a research of the best hyperparameters for a model.

This feature is executed by launching the command:

``python experimentation/hyperparameter_search.py {stopwords} {normalization} {vectorizer} {model_selected}``

where :
- stopwords : 
    - **0** :  To not remove the stop words.
    - "**english**" : To remove english stop words. nltk stop words will be used.

- normalization :
    - **0** : To not perform normalization.
    - **1** : To perform lemmatizing. The **WordNetLemmatizer** from nltk will be used.
    - **2** : To perform stemming. The **PorterStemmer** from nltk will be used.

- vectorizer :
    - **1** : To use count vectorizer.
    - **2** : To use a tfidf vectorizer.

- model_selected : is the id of the model to fine tune. The id are the same as those of the model_id from **simple validation** and **cross validation** features. (we advise to not fine tune all model at once due to high computation time).

All the fine tuning results will be save to ``result_analysis/results/hyperparameter_search``

## Size evaluation

To evaluate how the model performs depending of the **training size** or the **number of dimension**. 

This feature is executed by launching the command:

 ``python experimentation/size_evaluation.py {stopwords} {normalization} {vectorizer} {new_experiment}  {experiment}  {evaluation} {range_step}``

 where :
 - stopwords, normalization, vectorizer, new_experiment and  experiment are the same as the **simple validation** and **cross validation** features.

 - evaluation :
    - **1** : To evaluate the training size.
    - **2** : To evaluate the dimensionality size.
 - range_step :
    - an integer, the step to increase the size. for example if 1000 and if max size is 12000. a cross validation will be perform with a size (training or dimemnsionality size) from 1000 to 12000 with a step of 1000.



