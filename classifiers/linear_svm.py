import pandas as pd
from sklearn.svm import LinearSVC, SVC

from classifiers.base_classifier import estimate


def run(X_cols_add: [str] = [],
        y_col: str = 'grade_normalized',
        df=pd.read_csv('data/dataset_preprocessed_with_grade.csv'),
        ):
    classifier = LinearSVC(C=0.025, verbose=True, dual=False, max_iter=1700, multi_class='ovr')
    return estimate(df, classifier, X_cols_add, y_col, 'svc_linear')
