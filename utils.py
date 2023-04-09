import numpy as np
import yaml
from sklearn.preprocessing import MinMaxScaler


def read_config():
    with open('config.yaml', 'r') as stream:
        return yaml.safe_load(stream)


def normalize_columns(df, columns_to_normalize):
    for column in columns_to_normalize:
        df[column + '_normalized'] = MinMaxScaler().fit_transform(np.array(df[column]).reshape(-1, 1))
