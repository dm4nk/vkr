import pandas as pd
from xgboost import XGBClassifier

from classifiers.base_classifier import estimate


def run(X_cols_add: [str] = [],
        y_col: str = 'log1p_likes_normalized',
        df=pd.read_csv('data/dataset_preprocessed_5_percent.csv'),
        ):
    xgboost = XGBClassifier(n_estimators=1300, max_depth=None, objective='reg:squarederror')

    return estimate(df, xgboost, X_cols_add, y_col, 'xgboost')
