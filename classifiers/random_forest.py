import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from classifiers.base_classifier import estimate


def run():
    X_cols_add = [col + '_normalized' for col in
                  ['len_text', 'hashtag_count', 'url_count', 'emoji_count', 'time_window_id', 'dayofweek',
                   'attachments']]

    y_col = 'log1p_likes_normalized'

    df = pd.read_csv('data/dataset_preprocessed_5_percent.csv')

    classifier = RandomForestClassifier(criterion='gini',
                                        min_samples_split=2,
                                        min_samples_leaf=2,
                                        max_features='sqrt',
                                        n_estimators=1591,
                                        max_depth=None,
                                        n_jobs=10,
                                        verbose=True)

    estimate(df, classifier, X_cols_add, y_col)


run()
