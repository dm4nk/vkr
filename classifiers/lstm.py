from datetime import datetime
from time import time

import pandas as pd
from keras.callbacks import ReduceLROnPlateau
from keras.layers import LSTM, Dense, Dropout, Input, Embedding, SimpleRNN
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from utils import read_config


def run(X_cols_add: [str] = [],
        y_col: str = 'log1p_likes_normalized',
        df=pd.read_csv('data/dataset_preprocessed_10_percent.csv'),
        ):
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
            like_to_category[like] = 1

    Y = df[y_col].apply(lambda like_row: like_to_category.get(like_row))

    le = LabelEncoder()
    Y = le.fit_transform(Y)
    Y = Y.reshape(-1, 1)
    print(Y)

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

    inputs = Input(shape=[max_len])
    layer = Embedding(max_words, 500, input_length=max_len)(inputs)
    layer = LSTM(64, return_sequences=True)(layer)
    layer = LSTM(64, return_sequences=True)(layer)
    layer = LSTM(64)(layer)
    layer = Dense(64, activation='relu')(layer)
    # layer = Dropout(0.2)(layer)
    # layer = Dense(8, activation='relu')(layer)
    # layer = Dense(3, name='out_layer', activation='softmax')(layer)
    layer = Dropout(0.1)(layer)
    layer = Dense(1, activation='sigmoid')(layer)

    model = Model(inputs=inputs, outputs=layer)
    model.summary()
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=0.0001),
                  metrics=['accuracy'])

    history = model.fit(sequences_matrix, Y_train, batch_size=1024, epochs=30, validation_split=0.2, callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.000001)])

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('График обучения нейронной сети (Точность)')
    plt.ylabel('Точность')
    plt.xlabel('Эпоха')
    plt.legend(['Тренировочная выборка', 'Валидационная выборка'], loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.savefig("img/accuracy.svg")
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('График обучения нейронной сети (Функция ошибок)')
    plt.ylabel('Функция ошибок')
    plt.xlabel('Эпоха')
    plt.legend(['Тренировочная выборка', 'Валидационная выборка'], loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.savefig("img/loss.svg")
    plt.show()

    test_sequences = tok.texts_to_sequences(X_test)
    test_sequences_matrix = pad_sequences(test_sequences, maxlen=max_len)
    print(test_sequences_matrix)

    y_pred = model.predict(test_sequences_matrix)
    print(y_pred)
    y_pred[y_pred <= 0.5] = 0
    y_pred[y_pred > 0.5] = 1

    f1 = f1_score(y_pred, Y_test, average='macro')
    precision = precision_score(y_pred, Y_test, average='macro')
    recall = recall_score(y_pred, Y_test, average="macro")
    accuracy = accuracy_score(y_pred, Y_test)

    end = time()
    log_str_results = f'executed {(end - start) / 60:.2f} minutes\n' + \
                      f'F1: {f1 * 100:.2f}\n' \
                      f'Precision: {precision * 100:.2f}\n' \
                      f'Recall: {recall * 100:.2f}\n' \
                      f'Accuracy: {accuracy * 100:.2f}'
    print(log_str_results)

    with open(f'logs/lstm.txt', 'a') as classifier_log:
        classifier_log.write(log_str + log_str_results)


run()
