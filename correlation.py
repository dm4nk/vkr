import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def run():
    df = pd.read_csv("data/1_try/dataset.csv")
    df = df.drop(df[df.domain != 'rentvchannel'].index)
    domains = df.domain.unique()
    domains_dict = dict(zip(domains, range(len(domains))))
    df['domain_id'] = df.domain.map(domains_dict)
    plt.figure(figsize=(21, 21))
    sns.heatmap(df.corr(), center=0, annot=True, fmt='.1%', cmap="PiYG")
    plt.show()


run()
