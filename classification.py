import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from utils import read_config


def run():
    X_cols_add = []
    y_col = 'log1p_likes_normalized'

    config = read_config()
    bad = config['classification']['ranges']['bad']
    good = config['classification']['ranges']['good']
    best = config['classification']['ranges']['best']

    ngram_range = (config['classification']['ngram']['min'], config['classification']['ngram']['max'])
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_df=0.90, min_df=0.001)

    df = pd.read_csv('data/dataset_preprocessed.csv')
    df.drop(df[df.text == ''].index, inplace=True)
    df.dropna(inplace=True)
    X = df[['preprocessed_text'] + X_cols_add]
    X_corpus = vectorizer.fit_transform(X['preprocessed_text']).toarray()
    X_full = np.c_[X_corpus, X[X_cols_add].values]
    X_features = vectorizer.get_feature_names_out()

    print(len(X_features))

    size = df.shape[0]
    counter = 0
    like_to_category = {}

    for like, count in df.groupby(y_col)[y_col].count().items():
        counter += count
        if counter / size < bad:
            like_to_category[like] = 0
        elif counter / size < bad + good:
            like_to_category[like] = 1
        else:
            like_to_category[like] = 2

    y = df[y_col].apply(lambda like_row: like_to_category.get(like_row))

    X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=0.2, random_state=1)

    rf = RandomForestClassifier(n_estimators=1000, verbose=True)
    rf.fit(X_train, y_train)

    predictions = rf.predict(X_test)
    cm = confusion_matrix(y_test, predictions, labels=rf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf.classes_)
    disp.plot()
    plt.show()


run()
