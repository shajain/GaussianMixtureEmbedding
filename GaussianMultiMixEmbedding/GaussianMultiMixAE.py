import tensorflow as tf
import numpy as np
from scipy.stats import bernoulli
from KDE.kde import bandwidth
from KDE.kde import kde
from AutoEncoder.AutoEncoder import AutoEncoderBasic
import tensorflow_probability as tfp
#import tensorflow_probability as tfp
from GaussianEmbedding.GaussianAE import GaussianAE
from GaussianMixEmbedding.GaussianMixAE import GaussianMixAE
from NN.models import BasicMultiClass as RNet
from itertools import chain
import pdb


class GaussianMultiMixGMMAE(AutoEncoderBasic):

    def __init__(self, input_dim=1, latent_dim=1, nComps=2, nMix=2, cMemPerSample=None, **kwargs):
        # membership is a list of numpy vectors that tells which components are present in which sample.
        # The ith vector in the list contains the index of components present in that mixture.
        super(GaussianMultiMixGMMAE, self).__init__(input_dim, latent_dim, **kwargs)
        self.nComps = nComps
        self.nMix = nMix
        self.cMemPerSample = cMemPerSample
        if self.cMemPerSample is None:
            self.cMemPerSample = [np.arange(nComps) for i in np.arange(nMix)]

        self.cMem2SMem(self.cMemPerSample)

        self.gMixAE = []
        if 'gMixAE' in kwargs:
            self.gMixAE = kwargs['gMixAE']
        else:
            kwargs = {'encoder': self.encoder, 'decoder': self.decoder}
            self.gMixAE = [GaussianMixAE(input_dim, latent_dim, len(mem), **kwargs) for mem in self.cMemPerSample]


    def cMem2SMem(self, membership):
        self.sMemPerComp = [[] for _ in range(self.nComps)]
        self.cIndPerComp = [[] for _ in range(self.nComps)]
        # Iterate over the membership variable and update the component membership and indexes variables
        for i, mix in enumerate(membership):
            for j, component in enumerate(mix):
                self.sMemPerComp[component].append(i)
                self.cIndPerComp[component].append(j)

    def testGradient(self, data, batchSize):
        self.randomUniform()
        self.bw = bandwidth(batchSize)
        x = data['x']
        n = x.shape[0]
        ix = np.random.choice(n, batchSize, replace=True)
        x = x[ix, :]
        enc = self.normalized_encoding(x)
        xx = tf.constant([[0.0,8.0, 9.0], [0.0, 5.0, 6.0]])
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            # pdb.set_trace()
            tape.watch(xx)
            loss = tf.reduce_sum(self.phiDTF(xx))
        g = tape.gradient(loss, xx)
        pdb.set_trace()
        return


    def lossTF(self, X):
        loss_gae = tf.add_n([ae.lossTF(x) for (ae, x) in zip(self.gMixAE, X)])
        loss_consistency =self.consistencyLossTF(X)
        loss = loss_gae + loss_consistency
        return loss

    def randomUniform(self):
        [gae.randomUniform() for gae in self.gMixAE]

    def subSample(self, X, batchSize):
        X = [gae.subSample(x, batchSize) for (x, gae) in zip(X,self.gMixAE)]
        return X

    def updateBandwidth(self, X):
        [gae.updateBandwidth(x) for (x, gae) in zip(X, self.gMixAE)]

    def gradients(self, data, batchSize):
        #pdb.set_trace()
        self.randomUniform()
        X = self.subSample(data['X'], batchSize)
        self.updateBandwidth(X)
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            # pdb.set_trace()
            tape.watch(self.getTrainableVariables())
            loss = self.lossTF(X)
        return loss, tape.gradient(loss, self.getTrainableVariables())




    def loss(self, data, R=None):
        X = data['X']
        L = [ae.loss(x) for (ae, x) in zip(self.gMixAE, X)]
        Loss_gae = [l[0] for l in L]
        loss_gae = sum(Loss_gae)
        LossRec = [l[1] for l in L]
        LossGaussian = [l[2] for l in L]
        loss_consistency, muLosses, stdLosses = self.consistencyLoss(X)
        loss = loss_gae + loss_consistency
        return loss, loss_consistency, LossRec, LossGaussian, muLosses, stdLosses

    def getTrainableVariables(self):
        #vars = self.rNet.trainable_variables + super(GaussianMMixAE, self).getTrainableVariables()
        #pdb.set_trace()
        vars = [l.deref() for l in set([l.ref() for ae in self.gMixAE for l in ae.getTrainableVariables()])]
        return vars


    def copy(self):
        gMixAE = [ae.copy() for ae in self.gMixAE]
        encoder = gMixAE[0].encoder
        decoder = gMixAE[0].decoder
        for gae in gMixAE:
            gae.encoder = encoder
            gae.decoder = decoder
        kwargs = {'input_dim': self.input_dim, 'latent_dim': self.latent_dim, 'nComps': self.nComps,
                  'nMix': self.nMix, 'membership': self.cMemPerSample,
                  'encoder': encoder, 'decoder': decoder, 'gMixAE': gMixAE}
        ae = type(self)(**kwargs)
        # ae.latent_dim = self.latent_dim
        # ae.input_dim = self.input_dim
        return ae