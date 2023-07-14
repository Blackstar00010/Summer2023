import numpy as np
from scipy.stats import multivariate_normal

# DEBUG = True

def debug(*args, **kwargs):
    global DEBUG
    if DEBUG:
        print(*args, **kwargs)


#
def phi(Y, mu_k, cov_k):
    """
    TODO
    :param Y:
    :param mu_k:
    :param cov_k:
    :return: dataframe of musigi
    """
    norm = multivariate_normal(mean=mu_k, cov=cov_k, allow_singular=True)
    return norm.pdf(Y)


# 가우시안 분포가 data를 설명하는 정도의 책임값 계산.
def getExpectation(Y, mu, cov, alpha):
    N = Y.shape[0]
    K = alpha.shape[0]
    assert N > 1, "There must be more than one sample!"
    assert K > 1, "There must be more than one gaussian model!"
    gamma = np.mat(np.zeros((N, K)))
    prob = np.zeros((N, K))  
    for k in range(K):
        prob[:, k] = phi(Y, mu[k], cov[k])  
    prob = np.mat(prob)
    for k in range(K):
        gamma[:, k] = alpha[k] * prob[:, k]     
    for i in range(N):
        gamma[i, :] /= np.sum(gamma[i, :])  
    return gamma


# 책임값에 따라 평균, 공분산, 혼합계수 최적화.
def maximize(Y, gamma):
    N, D = Y.shape
    K = gamma.shape[1]
    mu = np.zeros((K, D))
    cov = []
    alpha = np.zeros(K)
    for k in range(K):
        Nk = np.sum(gamma[:, k])
        mu[k, :] = np.sum(np.multiply(Y, gamma[:, k]), axis=0) / Nk
        cov_k = (Y - mu[k]).T * np.multiply((Y - mu[k]), gamma[:, k]) / Nk
        cov.append(cov_k)
        alpha[k] = Nk / N
    cov = np.array(cov)
    return mu, cov, alpha


# data 0, 1 사이로 scale.
def scale_data(Y):
    for i in range(Y.shape[1]):
        max_ = Y[:, i].max()
        min_ = Y[:, i].min()
        Y[:, i] = (Y[:, i] - min_) / (max_ - min_)
    debug("Data scaled.")
    return Y


# initial parameter(평균, 공분산, 혼합계수) 무작위 지정.
def init_params(shape, K):
    N, D = shape
    mu = np.random.rand(K, D)
    cov = np.array([np.eye(D)] * K)
    alpha = np.array([1.0 / K] * K)
    debug("Parameters initialized.")
    debug("mu:", mu, "cov:", cov, "alpha:", alpha, sep="\n")
    return mu, cov, alpha


# 같은 시행을 변화가 없을 때까지 times 만큼 반복.
def GMM_EM(Y, K, times):
    Y = scale_data(Y)
    mu, cov, alpha = init_params(Y.shape, K)
    for i in range(times):
        gamma = getExpectation(Y, mu, cov, alpha)
        mu, cov, alpha = maximize(Y, gamma)
    debug("{sep} Result {sep}".format(sep="-" * 20))
    debug("mu:", mu, "cov:", cov, "alpha:", alpha, sep="\n")
    return mu, cov, alpha
