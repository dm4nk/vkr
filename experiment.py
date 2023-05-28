import pandas as pd

from classifiers import linear_svm, xgboost_forest, rbf_svm

X_cols_add = [col + '_normalized' for col in
              ['len_text', 'hashtag_count', 'url_count', 'emoji_count', 'time_window_id', 'dayofweek', 'attachments']]
y_cols = ['log1p_likes_normalized', 'log1p_comments_normalized', 'log1p_views_normalized', 'log1p_reposts_normalized']

df = pd.read_csv('data/dataset_preprocessed_50_percent.csv')


# df = pd.read_csv('data/dataset_preprocessed_with_semantic_bi.csv').sample(frac=0.5)

# df2 = pd.read_csv('data/dataset_preprocessed_50_percent.csv')
# df = pd.merge(df1, df2, how='inner', on=['date', 'likes', 'reposts', 'comments', 'views'])
# print(df.head())

# X_cols_add = [col + '_normalized' for col in
#                   ['len_text', 'hashtag_count', 'url_count', 'emoji_count', 'time_window_id', 'dayofweek',
#                    'attachments']] + ['neutral', 'positive', 'negative', 'skip', 'speech']
#
# df1 = pd.read_csv('data/dataset_preprocessed_with_grade.csv')
# df2 = pd.read_csv('data/dataset_preprocessed_with_semantic_uni.csv')[['neutral', 'positive', 'negative', 'skip', 'speech']]
# df = df1.join(df2).sample(frac=0.5)


def experiment_cascade_columns(classifier, filename):
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


def experiment_single_column(classifier, filename):
    w, h = len(y_cols) + 1, len(X_cols_add) + 1
    results = [['N/A' for x in range(h)] for y in range(w)]

    results[0][1] = str(X_cols_add)

    for j in range(len(y_cols)):
        results[j + 1][0] = y_cols[j]

    for j in range(len(y_cols)):
        results[j + 1][1] = classifier.run(X_cols_add, y_cols[j], df)

    print(results)
    pd.DataFrame(results).to_csv(f'results/iter2/{filename}.csv')


def experiment_two_columns(classifier, filename):
    w, h = len(y_cols) + 1, len(X_cols_add) + 1
    results = [['N/A' for x in range(h)] for y in range(w)]

    results[0][2] = str(X_cols_add)
    results[0][1] = 'No parameters'

    for j in range(len(y_cols)):
        results[j + 1][0] = y_cols[j]

    for j in range(len(y_cols)):
        results[j + 1][1] = classifier.run([], y_cols[j], df)

    for j in range(len(y_cols)):
        results[j + 1][2] = classifier.run(X_cols_add, y_cols[j], df)

    print(results)
    pd.DataFrame(results).to_csv(f'results/iter2/{filename}.csv')


# experiment_single(linear_svm, 'linear_svm_semantic_tri')
# experiment_single(random_forest, 'random_forest_semantic_tri')

# experiment_add(linear_svm, 'linear_svm_semantic_bi')
# experiment(linear_svm, 'linear_svm_semantic_bi')
# experiment(random_forest, 'random_forest')
# experiment(naive_bayes, 'naive_bayes')
# experiment_add(poly_svm, 'poly_svm')
# experiment_add(rbf_svm, 'rbf_svm')
# experiment_single(xbgoost_forest, 'xbgoost_forest_semantic_uni')

experiment_two_columns(linear_svm, 'linear_svm_two_columns')
# experiment_two_columns(xgboost_forest, 'xgboost_forest_two_columns')
