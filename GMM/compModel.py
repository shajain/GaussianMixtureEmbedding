import pdb

import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import multivariate_normal



class MVN():
    def __init__(self, mu=None, cov=None, spherical=False, unitCov=False):
        self.mu = mu
        self.cov = cov
        self.spherical = spherical
        self.unitCov = unitCov

    def copy(self):
        mu = np.copy(self.mu)
        cov = np.copy(self.cov)
        return MVN(mu, cov, self.spherical)

    def rvs(self, n):
        return multivariate_normal(self.mu, self.cov).rvs(n)

    def pdf(self, x):
        return multivariate_normal(self.mu, self.cov).pdf(x)

    def logpdf(self, x):
        #pdb.set_trace()
        return multivariate_normal(self.mu, self.cov).logpdf(x)

    def fit(self, X, W):
        #pdb.set_trace()
        if not isinstance(X, list):
            X = [X]
            W = [W]
        w_sum = sum([np.sum(w) for (x, w) in zip(X, W)])
        x_sum = sum([np.sum(x * w[:, None], axis=0) for (x, w) in zip(X, W)])
        self.mu = x_sum / w_sum
        #pdb.set_trace()
        Xbar = [x - self.mu for x in X]
        if not self.spherical and not self.unitCov:
            xtx_sum = sum([np.dot(w * (x.T), x) for (x, w) in zip(Xbar, W)])
            self.cov = xtx_sum / w_sum
        elif self.spherical:
            xSq_sum = sum([np.sum((x ** 2) * w[:, None], axis=0) for (x, w) in zip(Xbar, W)])
            self.cov = np.diag(xSq_sum) / w_sum
        else:
            self.cov = np.identity(self.mu.size)









