import keras.optimizers
from keras.layers import Dense, InputLayer
from keras.models import Sequential
import numpy as np
from keras import optimizers

def create_model(n_inputs, n_outputs):
    model = Sequential()
    model.add(InputLayer(input_shape=(n_inputs, )))
    model.add(Dense((n_inputs * 5) // 3, activation='relu', name='fc1'))
    model.add(Dense(n_outputs, activation='sigmoid', name='fc2'))

   # model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam())
    optimizer = optimizers.SGD(lr=0.01, momentum=0.9)  # Example: SGD optimizer
    model.compile(loss='binary_crossentropy', optimizer=optimizer)

    model.summary()
    return model


def train_model(model, features, labels, epoch_count, save_directory):
    model.fit(features, labels, epochs=epoch_count)
    model.save(save_directory)


def make_prediction(model, data):
    results = model.predict(np.asarray([data]))
    return results[0]

