from datetime import datetime
from time import time

import numpy as np
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from utils import read_config


def estimate(df: DataFrame,
             classifier,
             X_cols_add: [str],
             y_col: [str],
             name: str,
             max_df: float = 0.90,
             min_df: float = 0.005,
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

    counter = 0
    like_to_category = {}
    size = df.shape[0]
    for like, count in df.groupby(y_col)[y_col].count().items():
        counter += count
        if counter / size < bad:
            like_to_category[like] = 0
        elif counter / size < bad + good:
            like_to_category[like] = 1
        else:
            like_to_category[like] = 2

    y = df[y_col].apply(lambda like_row: like_to_category.get(like_row))

    X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=0.15)

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    f1 = f1_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average="macro")
    accuracy = accuracy_score(y_test, y_pred)

    end = time()

    log_str_results = f'executed {(end - start) / 60:.2f} minutes\n' + \
                      f'F1: {f1 * 100:.2f}\n' \
                      f'Precision: {precision * 100:.2f}\n' \
                      f'Recall: {recall * 100:.2f}\n' \
                      f'Accuracy: {accuracy * 100:.2f}'

    print(log_str_results)
    with open(f'logs/{name}.txt', 'a') as classifier_log:
        classifier_log.write(log_str + log_str_results)

    # cm = confusion_matrix(y_test, y_pred, labels=classifier.classes_)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
    # disp.plot()
    # plt.show()

    return log_str_results
