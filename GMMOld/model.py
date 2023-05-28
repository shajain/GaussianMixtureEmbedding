import pdb

import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import multivariate_normal
from GMMOld.estimates import  muEstimate, covEstimate, sphCovEstimate
from scipy.special import logsumexp
from sklearn.cluster import KMeans
from GMMOld.utils import responsibilityAndLogp
from sklearn.metrics import pairwise_distances


class UDist():
    def __init__(self, comps, pi):
        self.comps = comps
        self.pi = pi
        self.K = len(comps)

    def responsibility(self, x):
        #pdb.set_trace()
        return responsibilityAndLogp(x, self.comps, self.pi)[0]

    def copy(self):
        comps = [comp.copy() for comp in self.comps]
        return UDist(comps, np.copy(self.pi))

    def logLikelihood(self, x):
        logpdf = responsibilityAndLogp(x)[1]
        return np.mean(logpdf)

    def updatePar(self, x):
        r = self.responsibility(x)
        #pdb.set_trace()
        self.pi = np.mean(r, axis=0)
        print(self.pi)
        [self.comps[i].updatePar(x, r[:, i]) for i in np.arange(self.K)]

    def posterior(self, x):
        r = self.responsibility(x)
        ix_p = np.arange(np.floor((self.K/2))).astype('int64')
        p = np.sum(r[:, ix_p], axis=1)
        return p


    @classmethod
    def createObjWithGaussianPar(self, x, K, spherical=True):
        X = UDist.separateData(x, K)
        if spherical:
            comps = [SphGaussianComp.initPar(X[k]) for k in np.arange(K)]
        else:
            comps = [GaussianComp.initPar(X[k]) for k in np.arange(K)]
        pi = np.ones(K)/K
        return UDist(comps, pi)

    @classmethod
    def separateData(cls, x, K):
        kmeans = KMeans(n_clusters=K, init='k-means++').fit(x)
        X = [x[kmeans.labels_ == k, :] for k in np.arange(K)]
        return X



class PUDist(UDist):

    def __init__(self, comps, pi, ix_p, pi_p):
        super(PUDist, self).__init__(comps, pi)
        self.pi_p = pi_p
        self.ix_p = ix_p
        self.alpha = np.sum(self.pi[self.ix_p])


    def responsibility(self, x, x_p):
        r = super().responsibility(x)
        r_pos = responsibilityAndLogp(x_p, self.comps[self.ix_p], self.pi_p)[0]
        return r, r_pos

    def copy(self):
        comps = [comp.copy() for comp in self.comps]
        return PUDist(comps, np.copy(self.pi), np.copy(self.ix_p), np.copy(self.pi_p))

    def logLikelihood(self, x, x_p):
        ll = super().logLikelihood(x)
        logpdf_p = responsibilityAndLogp(x_p, self.comps[self.ix_p], self.pi_p)[1]
        ll_p = np.mean(logpdf_p)
        n = x.shape[0]
        n_p = x_p.shape[0]
        return (n*ll + n_p*ll_p)/(n + n_p)

    def updatePar(self, x, x_p):
        r, r_p = self.responsibility(x, x_p)
        self.pi = np.mean(r, axis=0)
        self.pi_p = np.mean(r_p, axis=0)
        self.alpha = np.sum(self.pi[self.ix_p])
        [self.comps[i].updatePar(x, r[:, i]) for i in np.arange(self.K) if i not in self.ix_p]
        xx_p = np.vstack(x, x_p)
        [self.comps[i].updatePar(xx_p, np.hstack(r[:, ix], r_p[:, i])) for (i, ix) in self.ix_p]

    def posterior(self, x):
        r = super().responsibility(x)
        p = np.sum(r[:, self.ix_p], axis = 1)
        return p


    @classmethod
    def createObjWithGaussianPar(self, x, x_p, K, spherical=True):
        X, ix_p = PUDist.separateData(x, x_p, K)
        if spherical:
            comps = [SphGaussianComp.initPar(X[k]) for k in np.arange(K)]
        else:
            comps = [GaussianComp.initPar(X[k]) for k in np.arange(K)]
        pi = np.ones(K) / K
        pi_p = np.ones_like(ix_p)/ix_p.shape[0]
        return PUDist(comps, pi, ix_p, pi_p)

    @classmethod
    def separateData(cls, x, x_p, K):
        kmeans_p = KMeans(n_clusters=np.floor(K/2), init='k-means++').fit(x_p)
        labels_p = kmeans_p.labels
        centers_p = kmeans_p.cluster_centers_
        kmeans = KMeans(n_clusters=K, init='k-means++').fit(x)
        labels = kmeans.labels
        centers = kmeans.cluster_centers_
        dist = pairwise_distances(centers_p, centers)
        ix_p = []
        for (i, drow) in enumerate(dist):
            drow[np.array(ix_p, dtype=int)] = np.inf
            ix = np.argsort(drow)
            ix_p.append(ix[0])
        ix_p = np.array(ix_p, dtype=int)
        X = [x[labels == k, :] for k in np.arange(K)]
        X[ix_p] = [np.vstack(X[ix], x_p[labels_p == i, :]) for (i, ix) in enumerate(ix_p)]
        return X, ix_p


class PNUDist(PUDist):

    def __init__(self, comps, pi, ix_p, pi_p, pi_n):
        super(PNUDist, self).__init__(comps, pi, ix_p, pi_p)
        self.pi_n = pi_n
        self.ix_n = np.setdiff1d(np.arange(self.K), self.ix_p)

    def responsibility(self, x, x_p, x_n):
        r, r_p = super().responsibility(x, x_p)
        r_n = responsibilityAndLogp(x_n, self.comps[self.ix_n], self.pi_n)[0]
        return r, r_p, r_n

    def copy(self):
        comps = [comp.copy() for comp in self.comps]
        return PUDist(comps, np.copy(self.pi), np.copy(self.ix_p), np.copy(self.pi_p), np.copy(self.pi_n))

    def logLikelihood(self, x, x_p, x_n):
        ll = super().logLikelihood(x, x_p)
        logpdf_n = responsibilityAndLogp(x_n, self.comps[self.ix_n], self.pi_n)[1]
        ll_n = np.mean(logpdf_n)
        n = x.shape[0]
        n_p = x_p.shape[0]
        n_n = x_n.shape[0]
        return ((n+n_p) * ll + n_n * ll_n) / (n + n_p + n_n)

    def updatePar(self, x, x_p, x_n):
        r, r_p, r_n = self.responsibility(x, x_p, x_n)
        self.pi = np.mean(r, axis=0)
        self.pi_p = np.mean(r_p, axis=0)
        self.pi_n = np.mean(r_n, axis=0)
        self.alpha = np.sum(self.pi[self.ix_p])
        xx_n = np.vstack(x, x_n)
        [self.comps[i].updatePar(xx_n, np.hstack(r[:, ix], r_n[:, i])) for (i, ix) in self.ix_n]
        xx_p = np.vstack(x, x_p)
        [self.comps[i].updatePar(xx_p, np.hstack(r[:, ix], r_p[:, i])) for (i, ix) in self.ix_p]

    def posterior(self, x):
        return super().posterior(x)


    @classmethod
    def createObjWithGaussianPar(self, x, x_p, x_n, K, spherical=True):
        X, ix_p, ix_n = PNUDist.separateData(x, x_p, x_n, K)
        if spherical:
            comps = [SphGaussianComp.initPar(X[k]) for k in np.arange(K)]
        else:
            comps = [GaussianComp.initPar(X[k]) for k in np.arange(K)]
        pi = np.ones(K) / K
        pi_p = np.ones_like(ix_p) / ix_p.shape[0]
        pi_n = np.ones_like(ix_n) / ix_n.shape[0]
        return PNUDist(comps, pi, ix_p, pi_p, pi_n)

    @classmethod
    def separateData(cls, x, x_p, x_n, K):
        kmeans_p = KMeans(n_clusters=np.floor(K / 2), init='k-means++').fit(x_p)
        labels_p = kmeans_p.labels
        centers_p = kmeans_p.cluster_centers_
        kmeans_n = KMeans(n_clusters=np.floor(K / 2), init='k-means++').fit(x_n)
        labels_n = kmeans_n.labels
        centers_n = kmeans_n.cluster_centers_
        kmeans = KMeans(n_clusters=K, init='k-means++').fit(x)
        labels = kmeans.labels
        centers = kmeans.cluster_centers_
        dist = pairwise_distances(centers_p, centers)
        ix_p = []
        for (i, drow) in enumerate(dist):
            drow[np.array(ix_p, dtype=int)] = np.inf
            ix = np.argsort(drow)
            ix_p.append(ix[0])
        ix_p = np.array(ix_p, dtype=int)
        dist = pairwise_distances(centers_n, centers)
        dist[:, ix_p] = np.inf
        ix_n = []
        for (i, drow) in enumerate(dist):
            drow[np.array(ix_n, dtype=int)] = np.inf
            ix = np.argsort(drow)
            ix_n.append(ix[0])
        ix_n = np.array(ix_n, dtype=int)
        X = [x[labels == k, :] for k in np.arange(K)]
        X[ix_n] = [np.vstack(X[ix], x_n[labels_n == i, :]) for (i, ix) in enumerate(ix_n)]
        X[ix_p] = [np.vstack(X[ix], x_p[labels_p == i, :]) for (i, ix) in enumerate(ix_p)]
        return X, ix_p, ix_n


class GaussianComp():
    def __init__(self, mu, cov):
        self.mu = mu
        self.cov = cov
        #print('cov', cov)
        self.rv = multivariate_normal(self.mu, self.cov)

    def copy(self):
        return GaussianComp(self.mu, self.cov)

    def pdf(self, x):
       return self.rv.pdf(x)

    def logpdf(self, x):
       return self.rv.logpdf(x)

    def updatePar(self, x, w):
        w_sum = np.sum(w)
        self.mu = muEstimate(x, w, w_sum)
        self.cov = covEstimate(x, w, w_sum, self.mu)

    @classmethod
    def initPar(cls, x):
        w_sum = x.shape[0]
        w = np.ones(w_sum)
        mu = muEstimate(x, w, w_sum)
        cov = covEstimate(x, w, w_sum, mu)
        return GaussianComp(mu, cov)


class SphGaussianComp(GaussianComp):
    def __init__(self, mu, var):
        super(SphGaussianComp, self).__init__(mu, np.diag(var))

    def copy(self):
        #pdb.set_trace()
        return SphGaussianComp(self.mu, np.diag(self.cov))

    def updatePar(self, x, w):
        w_sum = np.sum(w)
        self.mu = muEstimate(x, w, w_sum)
        self.cov = sphCovEstimate(x, w, w_sum, self.mu)
        #pdb.set_trace()

    @classmethod
    def initPar(cls, x):
        w_sum = x.shape[0]
        w = np.ones(w_sum)
        mu = muEstimate(x, w, w_sum)
        cov = sphCovEstimate(x, w, w_sum, mu)
        return SphGaussianComp(mu, cov)






