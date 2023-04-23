import pandas as pd
from sklearn.svm import LinearSVC

from classifiers.base_classifier import estimate


def run():
    X_cols_add = [col + '_normalized' for col in
                  ['len_text', 'hashtag_count', 'url_count', 'emoji_count', 'time_window_id', 'dayofweek',
                   'attachments']]

    y_col = 'log1p_likes_normalized'

    df = pd.read_csv('data/dataset_preprocessed.csv')

    classifier = LinearSVC(C=0.025, verbose=True, dual=False, max_iter=1700)
    estimate(df, classifier, X_cols_add, y_col)


run()
