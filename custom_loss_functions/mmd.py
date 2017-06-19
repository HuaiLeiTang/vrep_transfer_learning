import theano.tensor as T
import numpy as np


# Tensor calculation of MMD^2 given a sample X against the predictions of the network Y
# Quadratic time implementation
def mmd2_rbf_X_quad(X,sigma=1,biased=True):
    def mmd2_rbf_quad(y_true,Y):
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
    return mmd2_rbf_quad