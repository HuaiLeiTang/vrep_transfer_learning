from keras.layers import Input, Dense
from keras.models import Model
import keras
import matplotlib.pyplot as plt
import numpy as np
import theano.tensor as T
import keras.backend as K
import fisher_kernel as fk
from sklearn.mixture import GaussianMixture

def custom_mse2(y_true,y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)

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

def get_activations(model, layer, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], model.layers[layer].output)
    activations = get_activations([X_batch,0])
    return activations

N=32
data = np.random.normal(2,1,(N,2))
labels = data - 5

inputs = Input(shape=(2,))
hiddenlayer1 = Dense(100,kernel_regularizer = keras.regularizers.l2(0.01),activation='relu')(inputs)
hiddenlayer2 = Dense(100,kernel_regularizer = keras.regularizers.l2(0.01),activation='relu')(hiddenlayer1)
predictions = Dense(2)(hiddenlayer2)

model = Model(inputs=inputs, outputs=predictions)

K = 2
gmm = GaussianMixture(n_components=K, covariance_type='diag')
gmm.fit(data)
fv_X = fk.fisher_vector_np(data, gmm)

model.compile(optimizer='adam',
              loss=fk.mmd2_fisher(fv_X, gmm, N_=N, K_=K),
              metrics=['mse'])
model.fit(data, labels, batch_size=N, epochs=1000)
predicted = model.predict(data)

plt_x = plt.scatter(data[:,0],data[:,1],marker="+")
plt_y = plt.scatter(labels[:,0],labels[:,1],marker="o")
plt_pred = plt.scatter(predicted[:,0],predicted[:,1],marker="^")
plt.legend((plt_x,plt_y,plt_pred),
           ('Input','Labels','Predicted'))
plt.show()