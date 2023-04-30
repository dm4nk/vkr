import numpy as np
import optuna
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

from utils import read_config


def objective(trial, X, y):
    c = trial.suggest_float('svc_c', 1e-10, 1e10, log=True)
    max_iter = trial.suggest_int('max_iter', 1000, 5000, log=True)
    degree = trial.suggest_int('degree', 1, 5)
    cache_size = trial.suggest_int('cache_size', 200, 8000)

    clf = SVC(C=c, kernel='poly', degree=degree, max_iter=max_iter, cache_size=cache_size)

    return cross_val_score(clf, X, y, n_jobs=-1, cv=5).mean()


X_cols_add = []

y_col = 'log1p_likes_normalized'

config = read_config()
bad = config['classification']['ranges']['bad']
good = config['classification']['ranges']['good']
best = config['classification']['ranges']['best']

df = pd.read_csv('data/dataset_preprocessed_5_percent.csv')
X = df[['preprocessed_text'] + X_cols_add]
size = df.shape[0]
counter = 0
like_to_category = {}

for like, count in df.groupby(y_col)[y_col].count().items():
    counter += count
    if counter / size < bad:
        like_to_category[like] = 0
    # elif counter / size < bad + good:
    #     like_to_category[like] = 1
    else:
        like_to_category[like] = 1

vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_df=0.90, min_df=0.005)
X_corpus = vectorizer.fit_transform(X['preprocessed_text']).toarray()
X_full = np.c_[X_corpus, X[X_cols_add].values]

y = df[y_col].apply(lambda like_row: like_to_category.get(like_row))
study = optuna.create_study(study_name='Test study', direction='maximize')
study.optimize(lambda trial: objective(trial, X_full, y), n_trials=500)

trial = study.best_trial

print('Accuracy: {}'.format(trial.value))
print("Best hyperparameters: {}".format(trial.params))
