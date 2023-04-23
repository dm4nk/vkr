import re
from datetime import datetime
from time import time

import numpy as np
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from utils import read_config


def estimate(df: DataFrame, classifier, X_cols_add: [str], y_col: [str]):
    start = time()
    config = read_config()
    bad = config['classification']['ranges']['bad']
    good = config['classification']['ranges']['good']
    best = config['classification']['ranges']['best']

    ngram_range = (config['classification']['ngram']['min'], config['classification']['ngram']['max'])
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_df=0.90, min_df=0.005)

    df.fillna(0, inplace=True)
    X = df[['preprocessed_text'] + X_cols_add]
    X_corpus = vectorizer.fit_transform(X['preprocessed_text']).toarray()
    X_full = np.c_[X_corpus, X[X_cols_add].values]
    X_features = vectorizer.get_feature_names_out()
    size = df.shape[0]
    counter = 0
    like_to_category = {}

    log_str = f'\n\n{datetime.now()} {classifier.__class__.__name__} parameters\n' + \
              f'Features {len(X_features)}\n' + \
              f'X_cols_add: {X_cols_add}, y_col: {y_col}\n' + \
              f'Borders: bad({bad}) | good({good}) | best({best}))\n'
    print(log_str)

    for like, count in df.groupby(y_col)[y_col].count().items():
        counter += count
        if counter / size < bad:
            like_to_category[like] = 0
        # elif counter / size < bad + good:
        #     like_to_category[like] = 1
        else:
            like_to_category[like] = 2

    y = df[y_col].apply(lambda like_row: like_to_category.get(like_row))

    X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=0.15, random_state=2)

    classifier.fit(X_train, y_train, epochs=3, batch_size=1, verbose=2)
    y_pred = classifier.predict(X_test)

    f1 = f1_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average="macro")

    cm = confusion_matrix(y_test, y_pred, labels=classifier.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
    disp.plot()

    end = time()

    log_str_results = f'executed {(end - start) / 60} minutes\n' + \
                      f'F1: {f1}\nPrecision: {precision}\nRecall: {recall}'

    print(log_str_results)
    with open(f'logs/{re.sub(r"(?<!^)(?=[A-Z])", "_", classifier.__class__.__name__).lower()}.txt',
              'a') as classifier_log:
        classifier_log.write(log_str + log_str_results)

    plt.show()