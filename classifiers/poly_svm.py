import pandas as pd
from sklearn.svm import SVC

from classifiers.base_classifier import estimate


def run(X_cols_add: [str] = [],
        y_col: str = 'log1p_likes_normalized',
        df=pd.read_csv('data/dataset_preprocessed.csv'),
        ):

    classifier = SVC(C=0.3, kernel='poly', degree=2, max_iter=5000, cache_size=5012)
    estimate(df, classifier, X_cols_add, y_col, 'svc_poly')


run()
