from GaussianEmbedding.GaussianAE import GaussianAE as AutoEncoder
from NN.trainer import Trainer as Trainer
from GaussianEmbedding.debug import Debug
from misc.dictUtils import safeUpdate
from misc.dictUtils import safeRemove
from misc import sortedplot as sp
from scipy.stats import uniform
import numpy as np
import pdb


class GaussianEmbeddingFitting:

    autoEncoderDEF = {'n_units': 20, 'n_hidden': 10, 'dropout_rate': 0.2}
    #netDEF = {}
    trainDEF =  { 'batchSize': 500, 'maxIter': 1000, 'debug': False}

    def __init__(self, input_dim=1, latent_dim=1, **kwargs):
        self.autoEncoderDEF = safeUpdate(GaussianEmbeddingFitting.autoEncoderDEF, kwargs)
        self.autoEncoderDEF['input_dim'] = input_dim
        self.autoEncoderDEF['latent_dim'] = latent_dim
        self.trainDEF = safeUpdate(GaussianEmbeddingFitting.trainDEF, kwargs)
        self.autoEncoder = AutoEncoder(**self.autoEncoderDEF)
        #netPost.build((None, 1))

    def fit(self, x, **kwargs):
        data = {'x': x}
        self.fitArgs = {'data': data, **kwargs}
        trainer = Trainer(self.autoEncoder, data, **safeRemove(self.trainDEF, 'debug'))
        if self.trainDEF['debug']:
            self.debug = Debug()
            self.debug.attachData(data)
            # if 'posterior' in kwargs:
            #     self.debug.attachTarget(x, kwargs['posterior'])
            trainer.attachDebugger(self.debug)
        #pdb.set_trace()
        trainer.fit( )

    def getAutoEncoder(self):
        return self.autoEncoder

    def setNet(self, autoEncoder):
        self.autoEncoder = autoEncoder

    def refit(self):
        self.fit(**self.fitArgs)

    @classmethod
    def demo(cls):
        n = 2000
        x = np.hstack((uniform.rvs(size=(n, 1)),uniform.rvs(size=(n, 1))))
        #x = uniform.rvs(size=(n, 1))
        #pdb.set_trace()
        fitting = GaussianEmbeddingFitting(x.shape[1], 2, debug=True)
        fitting.fit(x)
        return fitting