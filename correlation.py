import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import locale

plt.rcParams['axes.formatter.use_locale'] = True
locale._override_localeconv["decimal_point"] = ','


def run():
    df = pd.read_csv("data/dataset_preprocessed_with_grade.csv")[['views', 'likes', 'reposts', 'comments', 'log1p_views_normalized', 'log1p_likes_normalized', 'log1p_reposts_normalized', 'log1p_comments_normalized',]]
    plt.figure(figsize=(30, 30))
    sns.heatmap(df.corr(numeric_only=True), center=0, annot=True, fmt='.2f', cmap="PiYG")
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    plt.show()


run()
