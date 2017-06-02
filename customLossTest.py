from keras.layers import Input, Dense
from keras.models import Model
import keras
import matplotlib.pyplot as plt
import numpy as np
import theano.tensor as T

def custom_mse(y_true,y_pred):
    return T.mean(T.square(y_pred - y_true), axis=-1)

def custom_mse_with_external_arg(arg):
    def loss(y_true,y_pred):
        return T.mean(T.square(y_pred - y_true), axis=-1)
    return loss

def rbf_mmd2(sigma=1,biased=True):
    def loss(X,Y):
        gamma = 1.0 / (2 * sigma**2)

        XX = T.dot(X, X.T)
        XY = T.dot(X, Y.T)
        YY = T.dot(Y, Y.T)

        X_sqnorms = T.diagonal(XX)
        Y_sqnorms = T.diagonal(YY)

        K_XY = T.exp(-gamma * (
                -2 * XY + X_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :]))
        K_XX = T.exp(-gamma * (
                -2 * XX + X_sqnorms[:, np.newaxis] + X_sqnorms[np.newaxis, :]))
        K_YY = T.exp(-gamma * (
                -2 * YY + Y_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :]))

        if biased:
            mmd2 = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
        else:
            m = K_XX.shape[0]
            n = K_YY.shape[0]

            mmd2 = ((K_XX.sum() - m) / (m * (m - 1))
                  + (K_YY.sum() - n) / (n * (n - 1))
                  - 2 * K_XY.mean())
        return mmd2
    return loss

N=50
data = np.random.normal(0,1,(N,2))
labels = np.random.normal(5,1,(N,2))

inputs = Input(shape=(2,))
hiddenlayer1 = Dense(200,kernel_regularizer = keras.regularizers.l2(0.01),activation='relu')(inputs)
hiddenlayer2 = Dense(200,kernel_regularizer = keras.regularizers.l2(0.01),activation='relu')(hiddenlayer1)
predictions = Dense(2)(hiddenlayer2)

model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop',
              loss=rbf_mmd2(),
              metrics=['mse'])
model.fit(data, labels, batch_size=4, epochs=30)
predicted = model.predict(data)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(data[:,0],data[:,1],marker="+")
ax.scatter(labels[:,0],labels[:,1],marker="o")
ax.scatter(predicted[:,0],predicted[:,1],marker="^")
