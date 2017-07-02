import numpy as np
from sklearn.mixture import GaussianMixture
import theano.tensor as T

def mmd2_fisher(fv_X, gmm):
    def mmd2(y_true, Y):
        T, D = Y.shape
        K = gmm.n_components
        Y2 = T.pow(Y, 2)

        # 1. Compute statistics
        S0 = T.zeros((K,))
        S1 = T.zeros((K, D))
        S2 = T.zeros((K, D))

        gamma = gmm.predict_proba(Y)  # posterior of the GMM
        gamma_S0 = gamma.sum(axis=0)  # sums up for all data points for each component

        S0 = S0 + gamma_S0
        for i in range(T):
            S1 += np.outer(gamma[i], Y[i])
            S2 += np.outer(gamma[i], Y2[i])

        # 2. Compute Fisher vector signatures
        sig_alpha = (S0 - (T * gmm.weights_)) / np.sqrt(gmm.weights_)
        sig_mean = T.zeros((K, D))
        sig_var = T.zeros((K, D))
        for i in range(K):
            sig_mean[i] += (S1[i] - gmm.weights_[i] * S0[i]) / (np.sqrt(gmm.weights_[i] * gmm.covariances_[i]))
            sig_var[i] += (S2[i] - 2 * gmm.weights_[i] * S1[i] + (gmm.weights_[i] ** 2 - gmm.covariances_[i] ** 2) *S0[i]) / \
                          (np.sqrt(gmm.weights_[i] * 2) * gmm.covariances_[i] ** 2)
        fv_Y = T.concatenate((sig_alpha, sig_mean.flatten(), sig_var.flatten()))

        # 3. Apply normalization
        fv_Y = T.sign(fv_Y) * T.absolute(fv_Y) ** 0.5
        fv_Y = fv_Y / T.sqrt(T.dot(fv_Y, fv_Y))

        fisher_kernel = T.dot(fv_X, fv_Y)

        return fisher_kernel

    return mmd2


def fisher_vector_np(x, gmm):
    """
    :param x: Datapoints (T datapoints with dimension D)
    :param gmm: Gaussian mixture model from sklearn (K components)

    :return fv: Normalized Fisher Vector representation (with dimension K * (2D + 1))

    Reference
    ---------
    Image Classification with the Fisher Vector: Theory and Practice
    J. Sanchez, F. Perronnin, T. Mensink, J. Verbeek
    """

    T, D = x.shape
    K = gmm.n_components
    x2 = x ** 2

    # 1. Compute statistics
    S0 = np.zeros((K,))
    S1 = np.zeros((K, D))
    S2 = np.zeros((K, D))

    gamma = gmm.predict_proba(x)  # posterior of the GMM
    gamma_S0 = gamma.sum(axis=0)  # sums up for all data points for each component

    S0 = S0 + gamma_S0
    for i in range(T):
        S1 += np.outer(gamma[i], x[i])
        S2 += np.outer(gamma[i], x2[i])

    # 2. Compute Fisher vector signatures
    sig_alpha = (S0 - (T * gmm.weights_)) / np.sqrt(gmm.weights_)
    sig_mean = np.zeros((K, D))
    sig_var = np.zeros((K, D))
    for i in range(K):
        sig_mean[i] += (S1[i] - gmm.weights_[i] * S0[i]) / (np.sqrt(gmm.weights_[i] * gmm.covariances_[i]))
        sig_var[i] += (S2[i] - 2 * gmm.weights_[i] * S1[i] + (gmm.weights_[i] ** 2 - gmm.covariances_[i] ** 2) * S0[i]) / \
                      (np.sqrt(gmm.weights_[i] * 2) * gmm.covariances_[i] ** 2)
    fv = np.concatenate((sig_alpha, sig_mean.flatten(), sig_var.flatten()))

    # 3. Apply normalization
    fv = np.sign(fv) * np.absolute(fv) ** 0.5
    fv = fv / np.sqrt(np.dot(fv, fv))

    return fv