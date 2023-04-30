import pandas as pd
from sklearn.naive_bayes import GaussianNB

from classifiers.base_classifier import estimate


def run(X_cols_add: [str] = [],
        y_col: str = 'log1p_likes_normalized',
        df=pd.read_csv('data/dataset_preprocessed.csv'),
        ):

    classifier = GaussianNB()
    estimate(df, classifier, X_cols_add, y_col, 'naive_bayes')


run()
