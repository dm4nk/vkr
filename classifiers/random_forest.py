import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from classifiers.base_classifier import estimate


def run(X_cols_add: [str] = [],
        y_col: str = 'log1p_likes_normalized',
        df=pd.read_csv('data/dataset_preprocessed.csv'),
        ):

    classifier = RandomForestClassifier(criterion='gini',
                                        min_samples_split=2,
                                        min_samples_leaf=2,
                                        max_features='sqrt',
                                        n_estimators=300,
                                        max_depth=None,
                                        n_jobs=2,
                                        verbose=True)

    estimate(df, classifier, X_cols_add, y_col, 'random_forest')


run()
