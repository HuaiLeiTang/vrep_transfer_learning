import numpy as np
from sklearn.mixture import GaussianMixture

def fisher_vector(x, gmm):
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
    x2 = x**2

    # 1. Compute statistics
    S0 = np.zeros((K,))
    S1 = np.zeros((K, D))
    S2 = np.zeros((K, D))

    gamma = gmm.predict_proba(x)     # posterior of the GMM
    gamma_S0 = gamma.sum(axis=0)     # sums up for all data points for each component

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
        sig_var[i]  += (S2[i] - 2*gmm.weights_[i]*S1[i] + (gmm.weights_[i]**2 - gmm.covariances_[i]**2)*S0[i]) / \
                       (np.sqrt(gmm.weights_[i]*2) * gmm.covariances_[i]**2)
    fv = np.concatenate((sig_alpha, sig_mean.flatten(), sig_var.flatten()))

    # 3. Apply normalization
    fv = np.sign(fv) * np.absolute(fv) ** 0.5
    fv = fv / np.sqrt(np.dot(fv, fv))

    return fv

def test():
    feature_size = 8
    dataset1 = np.random.normal(0, 1, (512, feature_size))
    dataset2 = np.random.normal(0, 1, (512, feature_size))
    dataset3 = np.random.normal(5, 1, (512, feature_size))
    dataset4 = np.random.normal(7, 2, (512, feature_size))

    K = 64
    gmm = GaussianMixture(n_components=K, covariance_type='diag')
    gmm.fit(dataset1)

    fv1 = fisher_vector(dataset1, gmm)
    fv2 = fisher_vector(dataset2, gmm)
    fv3 = fisher_vector(dataset3, gmm)
    fv4 = fisher_vector(dataset4, gmm)

    print "fv1: ", fv1
    print "fv2: ", fv2

    print "Fisher Kernel(d1,d1): ", np.dot(fv1, fv1)
    print "Fisher Kernel(d2,d2): ", np.dot(fv2, fv2)
    print "Fisher Kernel(d1,d2): ", np.dot(fv1, fv2)
    print "Fisher Kernel(d1,d3): ", np.dot(fv1, fv3)
    print "Fisher Kernel(d1,d4): ", np.dot(fv1, fv4)

if __name__=="__main__":
    # x = np.random.normal(0,1,(128,8))
    #
    # K = 32
    # gmm = GaussianMixture(n_components=K, covariance_type='diag')
    # gmm.fit(x)
    #
    # fv = fisher_vector(x, gmm)
    test()