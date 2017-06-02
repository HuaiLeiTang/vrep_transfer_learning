import numpy as np

N = 50
data1 = np.random.normal(0,1,(N,2))
data2 = np.random.normal(2,1,(N,2))
data3 = np.random.normal(5,10000,(N,2))
data4 = np.random.normal(0,1,(N,2))

def mmd2_np(X, Y, sigma=1, biased=True):
    gamma = 1.0 / (2 * sigma**2)

    XX = np.dot(X, X.T)
    XY = np.dot(X, Y.T)
    YY = np.dot(Y, Y.T)

    X_sqnorms = np.diag(XX)
    Y_sqnorms = np.diag(YY)

    K_XY = np.exp(-gamma * (
            -2 * XY + X_sqnorms[:,np.newaxis] + Y_sqnorms[np.newaxis,:]))
    K_XX = np.exp(-gamma * (
            -2 * XX + X_sqnorms[:,np.newaxis] + X_sqnorms[np.newaxis,:]))
    K_YY = np.exp(-gamma * (
            -2 * YY + Y_sqnorms[:,np.newaxis] + Y_sqnorms[np.newaxis,:]))

    if biased:
        mmd2 = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
    else:
        m = K_XX.shape[0]
        n = K_YY.shape[0]

        mmd2 = ((K_XX.sum() - m) / (m * (m - 1))
              + (K_YY.sum() - n) / (n * (n - 1))
              - 2 * K_XY.mean())
    return mmd2