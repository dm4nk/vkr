import pandas as pd
from keras.layers import Dense, LSTM
from keras.models import Sequential

from classifiers.base_classifier import estimate


def run():
    X_cols_add = [col + '_normalized' for col in
                  ['len_text', 'hashtag_count', 'url_count', 'emoji_count', 'time_window_id', 'dayofweek',
                   'attachments']]

    y_col = 'log1p_likes_normalized'

    df = pd.read_csv('data/dataset_preprocessed_5_percent.csv')

    lstm_model = Sequential()
    lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(993, 1)))
    lstm_model.add(LSTM(units=50))
    lstm_model.add(Dense(1))
    lstm_model.compile(loss='mean_squared_error', optimizer='adadelta')

    estimate(df, lstm_model, X_cols_add, y_col)


run()
