from MultiMixGMMEmbedding.MultiMixGMMAE import MultiMixGMMAE as AutoEncoder
from NN.trainer import Trainer as Trainer
from MultiMixGMMEmbedding.visualize import Visualizer as Debug
from misc.dictUtils import safeUpdate
from misc.dictUtils import safeRemove
from misc import sortedplot as sp
from scipy.stats import uniform
from scipy.stats import skewnorm
from scipy.stats import multivariate_normal as mvn
from data.distributions import mixture
from data.datagen import DataGenerator
from data.randomParameters import NormalMixPNParameters2
import numpy as np
import copy as cp
import pdb


class GMEFitting:

    autoEncoderDEF = {'n_units': 20, 'n_hidden': 10, 'dropout_rate': 0.2}
    #netDEF = {}
    trainDEF =  {'batchSize': 500, 'maxIter': 1000, 'debug': False}

    def __init__(self, input_dim=1, latent_dim=1, nComps=2, nMix=2, membership=None, **kwargs):
        self.autoEncoderDEF = safeUpdate(GMEFitting.autoEncoderDEF, kwargs)
        self.autoEncoderDEF['input_dim'] = input_dim
        self.autoEncoderDEF['latent_dim'] = latent_dim
        self.autoEncoderDEF['nComps'] = nComps
        self.autoEncoderDEF['nMix'] = nMix
        self.autoEncoderDEF['membership'] = membership
        self.trainDEF = safeUpdate(GMEFitting.trainDEF, kwargs)
        #pdb.set_trace()
        self.autoEncoder = AutoEncoder(**self.autoEncoderDEF)
        #netPost.build((None, 1))

    def fit(self, X, **kwargs):
        self.fitArgs = {'X': X, **kwargs}
        #pdb.set_trace()
        trainer = Trainer(self.autoEncoder, X, **safeRemove(self.trainDEF, 'debug'))
        if self.trainDEF['debug']:
            trainer.attachDebugger(self.debug)
        trainer.fit( )
    #
    # def getAutoEncoder(self):
    #     return self.autoEncoder
    #
    # def setNet(self, autoEncoder):
    #     self.autoEncoder = autoEncoder

    def refit(self):
        self.fit(**self.fitArgs)

    def initDebug(self, data, nComps, DG=None):
        self.debug = Debug(data, nComps, DG)

    @classmethod
    def demo(cls):
        n = 2000
        # x_n = norm.rvs(size=(n,1))
        # x_p = norm.rvs(size=(n, 1)) + 3
        nDims = 2
        nCompPerClass = 1
        parGen = NormalMixPNParameters2(nDims, nCompPerClass)
        irr_range = [0.01, 0.9, True]
        auc_pn = [0.9, 1]
        parGen.perturb2SatisfyMetrics(auc_pn, irr_range)
        dg = parGen.dg
        dg.alpha = 0.3
        x1 = dg.pn_data(n*2)[0]
        dg2 = cp.deepcopy(dg)
        dg2.alpha = 0.7
        x2 = dg2.pn_data(n * 2)[0]
        X = [x1, x2]
        #pdb.set_trace()

        DG = [dg, dg2]

        nComps = 2*nCompPerClass
        # pos_dist = skewnorm(-3, loc=2, scale=1)
        # neg_dist = skewnorm(3, loc=0, scale=1)
        # dg = DataGenerator(pos_dist, neg_dist, 0.5)
        #x = dg.pn_data(n*2)[0]
        # x = np.vstack(x_p,x_n)
        #pdb.set_trace()
        nMix = len(X)
        fitting = GMEFitting(nDims, nDims, nComps=nComps, nMix=nMix, debug=True)
        fitting.initDebug(X, nComps, DG)
        #fitting.initTrainer(data)
        fitting.fit(X)
        return fitting

    @classmethod
    def demoSingleCompSingleSample(cls):
        n = 2000
        # x_n = norm.rvs(size=(n,1))
        # x_p = norm.rvs(size=(n, 1)) + 3
        nDims = 2
        dist = mvn(mean=np.array([0.0,0.0]))
        x = dist.rvs(size=n)
        dg = mixture([dist], np.array([1.0]))
        X = [x]
        DG = [dg]
        nComps = 1
        nMix = len(X)
        fitting = GMEFitting(nDims, nDims, nComps=nComps, nMix=nMix, debug=True)
        fitting.initDebug(X, nComps, DG)
        #fitting.initTrainer(data)
        fitting.fit(X)
        return fitting