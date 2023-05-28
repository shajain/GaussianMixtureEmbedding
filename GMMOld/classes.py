import pdb

import numpy as np
from scipy.stats import norm
from scipy.stats import mvn
from sklearn.cluster import KMeans
from GMMOld.model import UDist, PUDist, PNUDist


class GMM:
    def __init__(self, K, dim, iter, spherical=False, **kwargs):
        self.K = K
        self.dim = dim
        self.iter = iter
        self.spherical = spherical
        if 'model' in kwargs:
            self.model = kwargs['model']
        else:
            self.model = None
        self.lls = []

    def initPar(self, x, x_p=None, x_n=None):
        if x_p is None:
            self.model = UDist.createObjWithGaussianPar(x, self.K, self.spherical)
        elif x_n is None:
            self.model = PUDist.createObjWithGaussianPar(x, x_p, self.K, self.spherical)
        else:
            self.model = PNUDist.createObjWithGaussianPar(x, x_p, x_n, self.K, self.spherical)

    def run(self, x, x_p=None, x_n=None):
        kwargs = {'x': x}
        if self.model is None:
            self.initPar(x, x_p, x_n)
        if x_p is not None:
            kwargs['x_p'] = x_p
        elif x_n is not None:
            kwargs['x_n'] = x_n
        [self.model.updatePar(**kwargs) for i in np.arange(self.iter)]

    def copy(self):
        kwargs = dict()
        #pdb.set_trace()
        if self.model is not None:
            kwargs['model'] = self.model.copy()
        return GMM(self.K, self.dim, self.iter, self.spherical, **kwargs)

