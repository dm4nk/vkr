import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def run():
    df = pd.read_csv("data/dataset_extended.csv")
    plt.figure(figsize=(30, 30))
    sns.heatmap(df.corr(numeric_only=True), center=0, annot=True, fmt='.1%', cmap="PiYG")
    plt.yticks(rotation=0)
    plt.show()


run()
