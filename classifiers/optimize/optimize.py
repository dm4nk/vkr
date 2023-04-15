import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score

from utils import read_config


def objective(trial, X_full, y):
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
    min_samples_split = trial.suggest_int('min_samples_split', 2, 50_000)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
    n_estimators = trial.suggest_int('n_estimators', 100, 2000)
    max_depth = int(trial.suggest_float('max_depth', 1, 2000, log=True))

    clf = RandomForestClassifier(n_estimators=n_estimators,
                                 max_depth=max_depth,
                                 criterion=criterion,
                                 min_samples_split=min_samples_split,
                                 min_samples_leaf=min_samples_leaf,
                                 max_features=max_features)

    return cross_val_score(clf, X_full, y, n_jobs=-1, cv=3).mean()


X_cols_add = [col + '_normalized' for col in
                  ['len_text', 'hashtag_count', 'url_count', 'time_window_id', 'dayofweek', 'attachments']]

y_col = 'log1p_likes_normalized'

config = read_config()
bad = config['classification']['ranges']['bad']
good = config['classification']['ranges']['good']
best = config['classification']['ranges']['best']

ngram_range = (config['classification']['ngram']['min'], config['classification']['ngram']['max'])
vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_df=0.90, min_df=0.01)

df = pd.read_csv('data/dataset_preprocessed.csv')
df.drop(df[df.text == ''].index, inplace=True)
df.dropna(inplace=True)
X = df[['preprocessed_text'] + X_cols_add]
X_corpus = vectorizer.fit_transform(X['preprocessed_text']).toarray()
X_full = np.c_[X_corpus, X[X_cols_add].values]
# X_features = vectorizer.get_feature_names_out()
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

# X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=0.15, random_state=1)
# classifier = trial.suggest_categorical('classifier', ['RandomForest', 'SVC'])

study = optuna.create_study(study_name='Test study', direction='maximize')
study.optimize(lambda trial: objective(trial, X_full, y), n_trials=1000)

trial = study.best_trial

print('Accuracy: {}'.format(trial.value))
print("Best hyperparameters: {}".format(trial.params))
