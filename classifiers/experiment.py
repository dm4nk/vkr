import pandas as pd

from classifiers import linear_svm, naive_bayes, poly_svm, random_forest, rbf_svm, xbgoost_forest

X_cols_add = [col + '_normalized' for col in
              ['len_text', 'hashtag_count', 'url_count', 'emoji_count', 'time_window_id', 'dayofweek', 'attachments',
               'domain_id']]
y_cols = ['log1p_likes_normalized', 'log1p_comments_normalized', 'log1p_views_normalized', 'log1p_reposts_normalized']

df = pd.read_csv('data/dataset_preprocessed_50_percent.csv')


def experiment(classifier, filename):
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


# experiment(linear_svm, 'linear_svm')
# experiment(random_forest, 'random_forest')
# experiment(naive_bayes, 'naive_bayes')
# experiment(poly_svm, 'poly_svm')
experiment(rbf_svm, 'rbf_svm')
experiment(xbgoost_forest, 'xbgoost_forest')
