from datetime import datetime
from time import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split

from utils import read_config


def estimate(df: DataFrame,
             classifier,
             X_cols_add: [str],
             y_col: [str],
             name: str,
             max_df: float = 0.90,
             min_df: float = 0.01,
             ):
    start = time()
    config = read_config()
    bad = config['classification']['ranges']['bad']
    good = config['classification']['ranges']['good']
    best = config['classification']['ranges']['best']

    ngram_range = (config['classification']['ngram']['min'], config['classification']['ngram']['max'])
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_df=max_df, min_df=min_df)

    X = df[['preprocessed_text'] + X_cols_add]
    X_corpus = vectorizer.fit_transform(X['preprocessed_text']).toarray()
    X_full = np.c_[X_corpus, X[X_cols_add].values]
    X_features = vectorizer.get_feature_names_out()
    log_str = f'\n\n{datetime.now()} {name} parameters\n' + \
              f'Features {len(X_features)}\n' + \
              f'X_cols_add: {str(X_cols_add)}, y_col: {y_col}\n' + \
              f'Borders: bad({bad}) | good({good}) | best({best}))\n'
    print(log_str)

    y = df[y_col]

    X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=0.15)

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    plt.scatter(y_test, y_pred)
    plt.scatter(y_test, y_test)

    end = time()

    log_str_results = f'executed {(end - start) / 60:.2f} minutes\n'

    print(log_str_results)
    with open(f'logs/{name}.txt', 'a') as classifier_log:
        classifier_log.write(log_str + log_str_results)

    plt.show()

    return log_str_results


def run(X_cols_add: [str] = [],
        y_col: str = 'log1p_likes_normalized',
        df=pd.read_csv('data/dataset_preprocessed_10_percent.csv'),
        ):
    def grade_scorer(y, y_pred, threshold: float = 0.001):
        condition = ((y < threshold) & (y < threshold)) | ((y >= threshold) & (y_pred >= threshold))
        coefs = np.where(condition, 0, 1)
        loss = np.abs(y - y_pred) * coefs
        return 1 / (loss.mean() + 1)

    classifier = RidgeCV(alphas=np.logspace(0, 1, num=10))
    return estimate(df, classifier, X_cols_add, y_col, 'ridge')


run()
