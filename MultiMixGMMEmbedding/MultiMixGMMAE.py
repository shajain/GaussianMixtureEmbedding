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
from GMM.gmm import GMM
from GMM.compModel import MVN
from GaussianLoss.CramerWoldLoss import CWLoss
from scipy.stats import uniform

from NN.models import BasicMultiClass as RNet
from itertools import chain
import pdb


class MultiMixGMMAE(AutoEncoderBasic):

    class SingleMixtureAE:
        def __init__(self, GMM, mmAE, sample_ix):
            self.GMM = GMM
            self.mmAE = mmAE
            self.sample_ix = sample_ix
        def responsibility(self, x):
            R = self.responsibilityFromEnc(self.mmAE.encoding(x))
            return R
        def responsibilityFromEnc(self, enc):
            R = self.GMM.__responsibility__(enc, self.sample_ix)[0]
            R = R.astype('float32')
            #pdb.set_trace()
            return R
        def normalized_encoding(self, enc):
            normEnc = [(enc - comp.mu)/np.sqrt(np.diag(comp.cov)) for comp in self.mixture().comps]
            #pdb.set_trace()
            #normEnc = [(enc - comp.mu) for comp in self.mixture().comps]
            return normEnc
        # def normalized_encodingTF(self, enc):
        #     normEnc = [(enc - comp.mu)/tf.stop_gradient(np.sqrt(np.diag(comp.cov))) for comp in self.mixture().comps]
        #     return normEnc
        def GaussianLoss(self, x):
            enc = self.mmAE.encoding(x)
            R = self.responsibilityFromEnc(enc)
            BW = [bandwidth(np.sum(r)) for r in R.T]
            normEnc = self.normalized_encoding(enc)
            losses_c = [self.mmAE.CWLoss.GaussianLoss(enc, r, bw) for (enc, r, bw) in zip(normEnc, R.T, BW)]
            loss = sum(losses_c)
            return loss, losses_c
        def GaussianLossTF(self, x, enc):
            #enc = tf.stop_gradient(self.mmAE.encoding(x))
            #pdb.set_trace()
            R = self.responsibilityFromEnc(enc)
            BW = [bandwidth(np.sum(r)) for r in R.T]
            normEnc = self.normalized_encoding(self.mmAE.encoder(x))
            loss = tf.add_n([self.mmAE.CWLoss.GaussianLossTF(enc, r, bw) for (enc, r, bw) in zip(normEnc, R.T, BW)])
            return loss

        def trueBatchSize(self, x, batchSize):
            n = x.shape[0]
            ix = np.random.choice(n, batchSize, replace=True)
            xx = x[ix, :]
            #pdb.set_trace()
            R = self.responsibility(xx)
            compProp = [np.mean(r) for r in R.T]
            trueBatchSize = np.floor(min(batchSize / min(compProp), min(n, 10*batchSize))).astype('int32')
            return trueBatchSize

        def subSample(self, x, batchSize):
            batchSize = self.trueBatchSize(x, batchSize)
            #pdb.set_trace()
            n = x.shape[0]
            ix = np.random.choice(n, batchSize, replace=True)
            x = x[ix, :]
            return x

        def mixture(self):
            return self.GMM.mixture[self.sample_ix]

        def kdeWrapper(self, x, batchSize=None, Orth=None):
            n = x.shape[0]
            if batchSize is None:
                batchSize = n
            x = self.subSample(x, batchSize)
            R = self.responsibility(x)
            BW = [bandwidth(np.sum(r)) for r in R.T]
            normEnc = self.normalized_encoding(x)
            IX = [bernoulli.rvs(r).astype('bool') for r in R.T]
            normEnc = [enc[ix, :] for (enc, ix) in zip(normEnc, IX)]
            if Orth is not None:
                normEnc = [np.matmul(enc, orth) for (enc, orth) in zip(normEnc, Orth)]

            KdeFncs = [[kde(enc[:, i:i+1], bw*np.std(enc[:,i])) for i in np.arange(self.mmAE.latent_dim)] for (enc,bw) in zip(normEnc, BW)]
            u = self.mmAE.randomUniform()
            probs = [[kdeFnc(u) for kdeFnc in kdeFncs] for kdeFncs in KdeFncs]
            #pdb.set_trace()
            return u, probs, normEnc



    def __init__(self, input_dim=1, latent_dim=1, nComps=2, nMix=2, cMemPerSample=None, **kwargs):
        # membership is a list of numpy vectors that tells which components are present in which sample.
        # The ith vector in the list contains the index of components present in that mixture.
        super(MultiMixGMMAE, self).__init__(input_dim, latent_dim, **kwargs)
        self.nComps = nComps
        self.nMix = nMix
        self.cMemPerSample = cMemPerSample
        if self.cMemPerSample is None:
            self.cMemPerSample = [np.arange(nComps) for i in np.arange(nMix)]
        #self.cMem2SMem(self.cMemPerSample)
        #pdb.set_trace()
        compDist = [MVN(spherical=True) for _ in range(nComps)]
        if 'GMM' not in kwargs:
            self.GMM = GMM(nComps, latent_dim, compDist=compDist, nMix=nMix, cMemPerSample=self.cMemPerSample)
        else:
            self.GMM = kwargs['GMM']
        self.CWLoss = CWLoss(dim=latent_dim)
        self.mixtureAEs = [MultiMixGMMAE.SingleMixtureAE(self.GMM, self, i) for i in range(nMix)]



    def lossTF(self, X, Enc):
        #pdb.set_trace()
        loss_recons = tf.add_n([self.reconstructionLossTF(x) for x in X])
        loss_gae = tf.add_n([mae.GaussianLossTF(x, enc) for (mae, x, enc) in zip(self.mixtureAEs, X, Enc)])
        loss = loss_gae + loss_recons
        #loss = loss_gae
        return loss

    def loss(self, X):
        loss_recons = sum([self.reconstructionLoss(x) for x in X])
        loss_gae = sum([mae.GaussianLoss(x) for (mae, x) in zip(self.mixtureAEs, X)])
        loss = loss_gae + loss_recons
        return loss, loss_recons, loss_gae


    def gradients(self, X, batchSize):
        self.GMM.refit([self.encoding(x) for x in X], 20)
        X = [mae.subSample(x, batchSize) for (mae, x) in zip(self.mixtureAEs, X)]
        Enc = [self.encoding(x) for x in X]
        #pdb.set_trace()
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            # pdb.set_trace()
            tape.watch(self.getTrainableVariables())
            loss = self.lossTF(X, Enc)
        return loss, tape.gradient(loss, self.getTrainableVariables())

    def randomUniform(self):
        self.u = uniform(loc=-4, scale=8).rvs(size=100)
        return self.u

    def copy(self):
        GMM = self.GMM.copy()
        encoder = self.copyEncoder()
        decoder = self.copyDecoder()
        kwargs = {'GMM': GMM, 'encoder': encoder, 'decoder': decoder}
        mmAE = MultiMixGMMAE(self.input_dim, self.latent_dim, self.nComps, self.nMix, self.cMemPerSample, **kwargs)
        return mmAE