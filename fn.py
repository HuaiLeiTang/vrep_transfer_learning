import theano.tensor as T
import keras.backend as K
import numpy as np

def empty(y_true,y_pred):
    return K.constant(0)

def rbf_mmd2_features(X,sigma=1,biased=True):
    def loss(y_true,Y):
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

def rbfmmd2_features(X,Y,sigma=1,biased=True):
    def loss(y_true,y_pred):
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

def mse_rbfmmd2(X,Y,sigma=1,biased=True):
    def loss(y_true,y_pred):
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
        return mmd2 + T.mean(T.square(y_pred - y_true), axis=-1)
    return loss

def get_activations(model, layer, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], model.layers[layer].output)
    activations = get_activations([X_batch,0])
    return activations