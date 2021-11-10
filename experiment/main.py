import pandas as pd



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, make_scorer
from experiment import Experiment



scorer = True


df = pd.read_csv('/Users/gonthierlucas/Desktop/DS_project/IMDB_reviews/data/IMDB_Dataset.csv')

X_train, X_test, y_train, y_test = train_test_split(df.review.to_numpy(), df.sentiment.apply(lambda x: 0 if (x == 'negative') else 1))


cv = CountVectorizer()

X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)
    

model = LogisticRegression()

if scorer:
    metrics =  {accuracy_score.__name__:make_scorer(accuracy_score), f1_score.__name__:make_scorer(f1_score),
     roc_auc_score.__name__:make_scorer((roc_auc_score))}
else :
    metrics = {accuracy_score.__name__:accuracy_score, f1_score.__name__:f1_score, roc_auc_score.__name__:roc_auc_score}

exp = Experiment('logisticRegression_cross', model)

exp.run_cross_valid_experimentation(X_train_cv, y_train,scorers=metrics, return_train_score=True)

  
