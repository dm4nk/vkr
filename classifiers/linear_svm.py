import pandas as pd
from sklearn.svm import LinearSVC

from classifiers.base_classifier import estimate


def run(X_cols_add: [str] = [],
        y_col: str = 'log1p_likes_normalized',
        df=pd.read_csv('data/dataset_preprocessed.csv'),
        ):
    classifier = LinearSVC(C=0.025, verbose=True, dual=False, max_iter=1700)
    return estimate(df, classifier, X_cols_add, y_col, 'svc_linear')
