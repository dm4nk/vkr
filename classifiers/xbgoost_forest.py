import pandas as pd
from xgboost import XGBClassifier

from classifiers.base_classifier import estimate


def run(X_cols_add: [str] = [],
        y_col: str = 'log1p_likes_normalized',
        df=pd.read_csv('data/dataset_preprocessed_5_percent.csv'),
        ):
    xbboost = XGBClassifier(n_estimators=300, max_depth=None, objective='binary:logistic')

    return estimate(df, xbboost, X_cols_add, y_col, 'xbboost')
