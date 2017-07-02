import numpy as np
from sklearn.mixture import GaussianMixture
import theano.tensor as T
import theano
import math

# import theano.configdefaults
# theano.configdefaults.config.exception_verbosity = 'high'


def fisher_vector_tensor(x, gmm):
    N, D = Y.shape
    K = gmm.n_components
    Y2 = T.sqr(Y)

    # 1. Compute statistics
    S0 = T.zeros((K,))
    S1 = T.zeros((K, D))
    S2 = T.zeros((K, D))

    #
    u = T.zeros((N, K))
    mean = gmm.means_
    cov_diag = gmm.covariances_
    w = gmm.weights_

    for n in range(N_):
        for k in range(K_):
            cov = T.nlinalg.diag(cov_diag[k])
            cov_inv = T.nlinalg.matrix_inverse(cov)
            cov_inv_diag = T.nlinalg.diag(cov_inv)
            u = T.set_subtensor(u[n, k],
                                1.0 / T.sqrt((2 * math.pi) ** D * T.prod(cov_diag[k])) *
                                T.exp(-0.5 * T.dot((Y[n] - mean[k]) * cov_inv_diag, Y[n] - mean[k])))

    wk_uk, updates = theano.scan(fn=lambda row, weights: row * weights,
                                 outputs_info=None,
                                 sequences=u,
                                 non_sequences=w)
    w_u_rowsum = T.sum(wk_uk, axis=1)
    gamma, updates = theano.scan(fn=lambda row, rowsum: row / rowsum,
                                 outputs_info=None,
                                 sequences=[wk_uk, w_u_rowsum])

    gamma_S0 = gamma.sum(axis=0)

    # gamma = gmm.predict_proba(Y)  # posterior of the GMM
    # gamma_S0 = gamma.sum(axis=0)  # sums up for all data points for each component

    S0 = S0 + gamma_S0
    for i in range(N_):
        S1 += T.outer(gamma[i], Y[i])
        S2 += T.outer(gamma[i], Y2[i])

    # 2. Compute Fisher vector signatures
    sig_alpha = (S0 - (N * gmm.weights_)) / np.sqrt(gmm.weights_)
    sig_mean = T.zeros((K, D))
    sig_var = T.zeros((K, D))
    for i in range(K_):
        sig_mean = T.inc_subtensor(sig_mean[i],
                                   (S1[i] - gmm.weights_[i] * S0[i]) / (np.sqrt(gmm.weights_[i] * gmm.covariances_[i])))
        sig_var = T.inc_subtensor(sig_var[i], (
        S2[i] - 2 * gmm.weights_[i] * S1[i] + (gmm.weights_[i] ** 2 - gmm.covariances_[i] ** 2) * S0[i]) / \
                                  (np.sqrt(gmm.weights_[i] * 2) * gmm.covariances_[i] ** 2))

    fv = T.concatenate((sig_alpha, sig_mean.flatten(), sig_var.flatten()))

    # 3. Apply normalization
    sign = T.where(fv < 0, -1.0, 1.0)
    fv = sign * T.abs_(fv) ** 0.5
    fv = fv / T.sqrt(T.dot(fv, fv))
    return fv

def mmd2_fisher(X, gmm, N_, K_):
    def mmd2(y_true, Y):

        N, D = Y.shape
        K = gmm.n_components
        Y2 = T.sqr(Y)

        # 1. Compute statistics
        S0 = T.zeros((K,))
        S1 = T.zeros((K, D))
        S2 = T.zeros((K, D))

        #
        u = T.zeros((N, K))
        mean = gmm.means_
        cov_diag = gmm.covariances_
        w = gmm.weights_

        for n in range(N_):
            for k in range(K_):
                cov = T.nlinalg.diag(cov_diag[k])
                cov_inv = T.nlinalg.matrix_inverse(cov)
                cov_inv_diag = T.nlinalg.diag(cov_inv)
                u = T.set_subtensor(u[n, k],
                                    1.0 / T.sqrt((2 * math.pi)**D * T.prod(cov_diag[k])) *
                                    T.exp(-0.5 * T.dot((Y[n] - mean[k]) * cov_inv_diag, Y[n] - mean[k])))

        wk_uk, updates = theano.scan(fn=lambda row, weights: row * weights,
                                     outputs_info=None,
                                     sequences=u,
                                     non_sequences=w)
        w_u_rowsum = T.sum(wk_uk, axis=1)
        gamma, updates = theano.scan(fn=lambda row, rowsum: row / rowsum,
                                     outputs_info=None,
                                     sequences=[wk_uk, w_u_rowsum])

        gamma_S0 = gamma.sum(axis=0)

        # gamma = gmm.predict_proba(Y)  # posterior of the GMM
        # gamma_S0 = gamma.sum(axis=0)  # sums up for all data points for each component

        S0 = S0 + gamma_S0
        for i in range(N_):
            S1 += T.outer(gamma[i], Y[i])
            S2 += T.outer(gamma[i], Y2[i])

        # 2. Compute Fisher vector signatures
        sig_alpha = (S0 - (N * gmm.weights_)) / np.sqrt(gmm.weights_)
        sig_mean = T.zeros((K, D))
        sig_var = T.zeros((K, D))
        for i in range(K_):
            sig_mean = T.inc_subtensor(sig_mean[i], (S1[i] - gmm.weights_[i] * S0[i]) / (np.sqrt(gmm.weights_[i] * gmm.covariances_[i])))
            sig_var = T.inc_subtensor(sig_var[i], (S2[i] - 2 * gmm.weights_[i] * S1[i] + (gmm.weights_[i] ** 2 - gmm.covariances_[i] ** 2) *S0[i]) / \
                          (np.sqrt(gmm.weights_[i] * 2) * gmm.covariances_[i] ** 2))
            # sig_mean[i] += (S1[i] - gmm.weights_[i] * S0[i]) / (np.sqrt(gmm.weights_[i] * gmm.covariances_[i]))
            # sig_var[i] += (S2[i] - 2 * gmm.weights_[i] * S1[i] + (gmm.weights_[i] ** 2 - gmm.covariances_[i] ** 2) *S0[i]) / \
            #               (np.sqrt(gmm.weights_[i] * 2) * gmm.covariances_[i] ** 2)
        fv_Y = T.concatenate((sig_alpha, sig_mean.flatten(), sig_var.flatten()))

        # 3. Apply normalization
        sign = T.where(fv_Y < 0, -1.0, 1.0)
        fv_Y = sign * T.abs_(fv_Y) ** 0.5
        fv_Y = fv_Y / T.sqrt(T.dot(fv_Y, fv_Y))

        return

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

if __name__=="__main__":
    N = 8
    D = 3
    dataset1 = np.random.normal(0, 1, (N, D))
    dataset2 = np.random.normal(0, 1, (N, D))
    dataset3 = np.random.normal(5, 1, (N, D))
    dataset4 = np.random.normal(7, 2, (N, D))

    K = 3
    gmm = GaussianMixture(n_components=K, covariance_type='diag')
    gmm.fit(dataset1)

    fv1 = fisher_vector_np(dataset1, gmm)
    fv2 = fisher_vector_np(dataset2, gmm)
    fv3 = fisher_vector_np(dataset3, gmm)
    fv4 = fisher_vector_np(dataset4, gmm)

    print "Fisher Kernel(d1,d1): ", np.dot(fv1, fv1)
    print "Fisher Kernel(d2,d2): ", np.dot(fv2, fv2)
    print "Fisher Kernel(d1,d2): ", np.dot(fv1, fv2)
    print "Fisher Kernel(d1,d3): ", np.dot(fv1, fv3)
    print "Fisher Kernel(d1,d4): ", np.dot(fv1, fv4)

    # Tensor test

    fisher_kernel = mmd2_fisher(fv1, gmm, N_=N, K_=K)