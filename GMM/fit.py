from GMM.gmm import GMM
from misc import sortedplot as sp
from data.randomParameters import NormalMixPNParameters2
import numpy as np
import copy as cp
from GMM.compModel import MVN
from GMM.debug import Debug
import pdb


class GMMFitting:
    def __init__(self, K, dim, maxIter=1000, compDist=None, nMix=1, cMemPerSample=None, debug=True):
        self.GMM = GMM(K, dim, maxIter, compDist, nMix, cMemPerSample)
        self.debug = True
    def fit(self, X, **kwargs):
        self.fitArgs = {'data': X, **kwargs}
        if self.debug:
            self.GMM.attachDebugger(self.debug)
        #pdb.set_trace()
        self.GMM.fit(X)

    def getAutoEncoder(self):
        return self.autoEncoder

    def setNet(self, autoEncoder):
        self.autoEncoder = autoEncoder

    def refit(self):
        self.fit(**self.fitArgs)

    def initDebug(self, X, dg=None):
        self.debug = Debug()
        self.debug.attachData(X, dg)

    @classmethod
    def demo(cls):
        n = 2000
        # x_n = norm.rvs(size=(n,1))
        # x_p = norm.rvs(size=(n, 1)) + 3
        nDims = 2
        nCompPerClass = 1
        parGen = NormalMixPNParameters2(nDims, nCompPerClass)
        irr_range = [0.01, 1]
        auc_pn = [0.9, 1]
        parGen.perturb2SatisfyMetrics(irr_range, auc_pn)
        dg = parGen.dg
        dg.alpha = 0.3
        x1 = dg.pn_data(n*2)[0]
        dg2 = cp.deepcopy(dg)
        dg2.alpha = 0.7
        x2 = dg2.pn_data(n * 2)[0]
        X = [x1, x2]

        DG = [dg, dg2]

        nMix = len(X)
        nComps = nCompPerClass*2
        cDist = [MVN(spherical=False) for _ in range(nComps)]
        cMemPerSample = [np.arange(nComps) for _ in range(nMix)]
        fitting = GMMFitting(nComps, nDims, compDist=cDist, nMix=nMix, cMemPerSample=cMemPerSample)
        fitting.initDebug(X, DG)
        fitting.fit(X)
        return fitting