from datetime import datetime
from time import time

import pandas as pd
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.models import Model
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from utils import read_config


def run():
    df = pd.read_csv('data/dataset_preprocessed.csv')

    X_cols_add = []
    y_col = 'log1p_likes_normalized'

    X = df.preprocessed_text

    start = time()
    config = read_config()
    bad = config['classification']['ranges']['bad']
    good = config['classification']['ranges']['good']
    best = config['classification']['ranges']['best']
    counter = 0
    like_to_category = {}
    size = df.shape[0]
    for like, count in df.groupby(y_col)[y_col].count().items():
        counter += count
        if counter / size < bad:
            like_to_category[like] = 0
        # elif counter / size < bad + good:
        #     like_to_category[like] = 1
        else:
            like_to_category[like] = 2

    Y = df[y_col].apply(lambda like_row: like_to_category.get(like_row))

    le = LabelEncoder()
    Y = le.fit_transform(Y)
    Y = Y.reshape(-1, 1)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15)

    log_str = f'\n\n{datetime.now()} LSTM parameters\n' + \
              f'Features {len(X)}\n' + \
              f'X_cols_add: {X_cols_add}, y_col: {y_col}\n' + \
              f'Borders: bad({bad}) | good({good}) | best({best}))\n'
    print(log_str)

    max_words = 1000
    max_len = 150
    tok = Tokenizer(num_words=max_words)
    tok.fit_on_texts(X_train)
    sequences = tok.texts_to_sequences(X_train)
    sequences_matrix = pad_sequences(sequences, maxlen=max_len)

    inputs = Input(name='inputs', shape=[max_len])
    layer = Embedding(max_words, 50, input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256, name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1, name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs, outputs=layer)
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

    model.fit(sequences_matrix, Y_train, batch_size=128, epochs=10,
              validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001)])

    test_sequences = tok.texts_to_sequences(X_test)
    test_sequences_matrix = pad_sequences(test_sequences, maxlen=max_len)
    accr = model.evaluate(test_sequences_matrix, Y_test)

    end = time()
    log_str_results = f'executed {(end - start) / 60} minutes\n' + \
                      f'Loss: {accr[0]}\nAccuracy: {accr[1]}\n'

    print(log_str_results)
    with open(f'logs/lstm.txt', 'a') as classifier_log:
        classifier_log.write(log_str + log_str_results)


run()
