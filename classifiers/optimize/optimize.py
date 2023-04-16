import numpy as np
import optuna
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC

from utils import read_config


def objective(trial, X, y):
    c = trial.suggest_float('svc_c', 1e-10, 1e10, log=True)
    ngram_range_max = trial.suggest_int('ngram_range_max', 1, 3)
    max_df = trial.suggest_float('max_df', 0.70, 1.0)
    min_df = trial.suggest_float('min_df', 0.005, 0.1)
    dual = trial.suggest_categorical('dual', [True, False])
    max_iter = trial.suggest_int('max_iter', 1000, 5000, log=True)

    ngram_range = (1, ngram_range_max)
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_df=max_df, min_df=min_df)
    X_corpus = vectorizer.fit_transform(X['preprocessed_text']).toarray()
    X_full = np.c_[X_corpus, X[X_cols_add].values]

    clf = LinearSVC(C=c, loss='squared_hinge', dual=dual, max_iter=max_iter)

    return cross_val_score(clf, X_full, y, n_jobs=-1, cv=5).mean()


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
        like_to_category[like] = 2

y = df[y_col].apply(lambda like_row: like_to_category.get(like_row))
study = optuna.create_study(study_name='Test study', direction='maximize')
study.optimize(lambda trial: objective(trial, X, y), n_trials=1000)

trial = study.best_trial

print('Accuracy: {}'.format(trial.value))
print("Best hyperparameters: {}".format(trial.params))
