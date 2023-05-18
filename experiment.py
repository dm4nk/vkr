import pandas as pd

from classifiers import random_forest, xbgoost_forest

X_cols_add = [col + '_normalized' for col in
              ['len_text', 'hashtag_count', 'url_count', 'emoji_count', 'time_window_id', 'dayofweek', 'attachments',
               'domain_id']]
y_cols = ['log1p_likes_normalized', 'log1p_comments_normalized', 'log1p_views_normalized', 'log1p_reposts_normalized']

df = pd.read_csv('data/dataset_preprocessed_50_percent.csv')


def experiment_add(classifier, filename):
    # TODO: обязательно добавить домен в исследования!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    w, h = len(y_cols) + 1, len(X_cols_add) + 1
    results = [['N/A' for x in range(h)] for y in range(w)]

    for i in range(len(X_cols_add)):
        results[0][i + 1] = X_cols_add[i]

    for j in range(len(y_cols)):
        results[j + 1][0] = y_cols[j]

    for i in range(len(X_cols_add)):
        for j in range(len(y_cols)):
            results[j + 1][i + 1] = classifier.run(X_cols_add[:i], y_cols[j], df)
    print(results)
    pd.DataFrame(results).to_csv(f'results/{filename}.csv')


def experiment_single(classifier, filename):
    w, h = len(y_cols) + 1, len(X_cols_add) + 1
    results = [['N/A' for x in range(h)] for y in range(w)]

    results[0][1] = str(X_cols_add)

    for j in range(len(y_cols)):
        results[j + 1][0] = y_cols[j]

    for j in range(len(y_cols)):
        results[j + 1][1] = classifier.run(X_cols_add, y_cols[j], df)

    print(results)
    pd.DataFrame(results).to_csv(f'results/{filename}.csv')


# experiment_single(linear_svm, 'linear_svm_semantic_tri')
# experiment_single(random_forest, 'random_forest_semantic_tri')

# experiment_add(linear_svm, 'linear_svm_semantic_bi')
# experiment(linear_svm, 'linear_svm_semantic_bi')
# experiment(random_forest, 'random_forest')
# experiment(naive_bayes, 'naive_bayes')
# experiment(poly_svm, 'poly_svm')
# experiment(rbf_svm, 'rbf_svm')


experiment_add(xbgoost_forest, 'xbgoost_forest')
