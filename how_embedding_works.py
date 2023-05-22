import numpy as np
from keras.models import Sequential
from keras.layers import Embedding

X = np.array([
    [0, 1],
    [1, 0]
])

model = Sequential()
model.add(Embedding(input_dim=2,
                    output_dim=2,
                    input_length=2))
print(model.get_weights())

model.compile(loss='mse', optimizer='sgd')

print(model.predict(X, batch_size=32))  # output (2, 5, 2)