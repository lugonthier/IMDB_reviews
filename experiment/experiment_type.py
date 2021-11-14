
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, make_scorer
from sklearn.model_selection import train_test_split


def launch_experiment(exp, mode, X, y, test_size=0.2):
    
    if mode == 1:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            metrics = {accuracy_score.__name__:accuracy_score, f1_score.__name__:f1_score, roc_auc_score.__name__:roc_auc_score}
            exp.run_simple_experimentation(X_train, y_train, X_test, y_test, "test_",  metrics)

    elif mode == 2:
        metrics =  {accuracy_score.__name__:make_scorer(accuracy_score), f1_score.__name__:make_scorer(f1_score),
        roc_auc_score.__name__:make_scorer((roc_auc_score))}

        exp.run_cross_valid_experimentation(X, y, scorers=metrics, return_train_score=True)


def training_size_evaluation(exp, df, mode, range_step):

    for size in range(range_step, len(df), range_step):
        X = df.review.to_numpy()[:size]
        y = df.sentiment.apply(lambda x: 0 if (x == 'negative') else 1).to_numpy()[:size]

        launch_experiment(exp, mode, X, y)


def dimensionality_size_evaluation(exp, df, mode, range_step):

    max_dim = 0
    
    for size in range(range_step, max_dim, range_step):

        pass