# Architecture 1: 3 Conv Layers, 16 filters each, RELU activation,
#                 1 FC layer with dropout=0.5

import h5py
import matplotlib.pyplot as plt
import keras
from keras.regularizers import l2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D


class Model:
    def __init__(self, input_shape, output_shape):
        model = Sequential()
        model.add(Conv2D(filters=16, kernel_size=(3, 3), strides=1,
                         padding='same', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(Conv2D(filters=16, kernel_size=(3, 3), strides=1,
                         padding='same', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(Conv2D(filters=16, kernel_size=(3, 3), strides=1,
                         padding='same', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512, kernel_regularizer=l2(0.01)))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(output_shape))
        self.model = model
        self.optimizer = keras.optimizers.adam(lr=0.001)
        self.history = None

    def compile(self):
        self.model.compile(loss='mean_squared_error',
                           optimizer=self.optimizer,
                           metrics=['mae'])

    def fit(self, x_train, y_train, x_test=None, y_test=None, batch_size=1, epochs=1):
        self.history = self.model.fit(x_train, y_train,
                                      batch_size=batch_size,
                                      epochs=epochs,
                                      validation_data=(x_test, y_test),
                                      shuffle="batch")

    def save(self, path, epochs):
        self.model.save(path)
        saved_model = h5py.File(path, "r+")
        for key in self.history.history.keys():
            dset = saved_model.create_dataset("history/"+key, (epochs,), dtype='f')
            dset[...] = self.history.history[key]
        saved_model.close()

    def plot(self):
        plt.plot(self.history.history['mean_absolute_error'])
        plt.plot(self.history.history['val_mean_absolute_error'])
        plt.title('Model MAE')
        plt.ylabel('mean absolute error')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()