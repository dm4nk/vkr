import pandas as pd
from sklearn.svm import SVC

from classifiers.base_classifier import estimate


def run(X_cols_add: [str] = [],
        y_col: str = 'log1p_likes_normalized',
        df=pd.read_csv('data/dataset_preprocessed_10_percent.csv'),
        ):
    classifier = SVC(C=0.85, kernel='rbf', gamma=1.0, max_iter=5000, cache_size=5012)
    return estimate(df, classifier, X_cols_add, y_col, 'rbf_svm')
