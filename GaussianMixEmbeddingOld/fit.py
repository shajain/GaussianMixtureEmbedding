import numpy as np

from GaussianMixEmbeddingOld.GaussianMixAE import GaussianMixAE as AutoEncoder
from NN.trainer import Trainer as Trainer
from GaussianMixEmbeddingOld.debug import Debug
from misc.dictUtils import safeUpdate
from misc.dictUtils import safeRemove
from data.randomParameters import NormalMixPNParameters2
from misc import sortedplot as sp
from scipy.stats import uniform
from scipy.stats import norm
from scipy.stats import skewnorm
from data.datagen import DataGenerator
import pdb


class GMEFitting:

    autoEncoderDEF = {'n_units': 20, 'n_hidden': 10, 'dropout_rate': 0.2}
    #netDEF = {}
    trainDEF =  {'batchSize': 500, 'maxIter': 1000}

    def __init__(self, input_dim=1, latent_dim=1, nComps=2, **kwargs):
        self.autoEncoderDEF = safeUpdate(GMEFitting.autoEncoderDEF, kwargs)
        self.autoEncoderDEF['input_dim'] = input_dim
        self.autoEncoderDEF['latent_dim'] = latent_dim
        self.autoEncoderDEF['nComps'] = nComps
        self.trainDEF = safeUpdate(GMEFitting.trainDEF, kwargs)
        self.autoEncoder = AutoEncoder(**self.autoEncoderDEF)
        self.debug = None
        #netPost.build((None, 1))

    def fit(self):
        self.trainer.fit( )

    def initDebug(self, data, model=None):
        self.debug = Debug()
        self.debug.attachData(data)
        if model is not None:
            self.debug.attachModel(model)

    def initTrainer(self, data):
        self.trainer = Trainer(self.autoEncoder, data, **self.trainDEF)
        if self.debug is not None:
            self.trainer.attachDebugger(self.debug)



    def getAutoEncoder(self):
        return self.autoEncoder

    def setNet(self, autoEncoder):
        self.autoEncoder = autoEncoder

    def refit(self):
        self.fit()

    @classmethod
    def demo(cls):
        n = 2000
        # x_n = norm.rvs(size=(n,1))
        # x_p = norm.rvs(size=(n, 1)) + 3
        # parGen = NormalMixPNParameters2(1, 1)
        # irr_range = [0.01, 1]
        # auc_pn = [0.9, 1]
        # parGen.perturb2SatisfyMetrics(irr_range, auc_pn)
        # dg = parGen.dg
        # dg.alpha = 0.5
        # x = dg.pn_data(n*2)[0]
        pos_dist = skewnorm(-3, loc=2, scale=1)
        neg_dist = skewnorm(3, loc=0, scale=1)
        dg = DataGenerator(pos_dist, neg_dist, 0.5)
        x = dg.pn_data(n*2)[0]
        # x = np.vstack(x_p,x_n)
        #pdb.set_trace()
        fitting = GMEFitting(x.shape[1], 1, 2, debug=True)
        data = {'x': x}
        fitting.initDebug(data, dg)
        fitting.initTrainer(data)
        fitting.fit()
        return fitting