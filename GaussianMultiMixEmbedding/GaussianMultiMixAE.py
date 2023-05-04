import tensorflow as tf
import numpy as np
from scipy.stats import bernoulli
from KDE.kde import bandwidth
from KDE.kde import kde
from AutoEncoder.AutoEncoder import AutoEncoderBasic
import tensorflow_probability as tfp
#import tensorflow_probability as tfp
from GaussianEmbedding.GaussianAE import GaussianAE
from GaussianMixEmbedding import GaussianMixAE
from NN.models import BasicMultiClass as RNet
from itertools import chain
import pdb


class GaussianMultiMixAE(AutoEncoderBasic):
    def __init__(self, input_dim=1, latent_dim=1, nComps=2, nMix=2, membership=None, **kwargs):
        super(GaussianMultiMixAE, self).__init__(input_dim, latent_dim, **kwargs)
        self.nComps = nComps
        self.nMix = nMix
        self.membership = membership
        if self.membership is None:
            self.membership = [np.arange(nComps) for i in np.arange(nMix)]

        self.gMixAE = []
        if 'gMixAE' in kwargs:
            self.gMixAE = kwargs['gMixAE']
        else:
            kwargs = {'encoder': self.encoder, 'decoder': self.decoder}
            self.gMixAE = [GaussianMixAE(input_dim, latent_dim, len(mem), **kwargs) for mem in self.membership]



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
        loss = tf.add_n([ae.lossTF(x) for (ae, x) in zip(self.gMixAE, X)])
        return loss

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


    def trueBatchSize(self, data, batchSize):
        x = data['x']
        n = x.shape[0]
        ix = np.random.choice(n, batchSize, replace=True)
        xx = x[ix, :]
        R = self.responsibility(xx)
        compProp = [np.mean(r) for r in R.T]
        trueBatchSize = np.floor(min(batchSize/min(compProp), n)).astype('int32')
        return trueBatchSize

    def subSample(self, X, batchSize):
        X = [ae.subSample(x, batchSize) for (ae, x) in zip(self.gMixAE, X)]
        return X

    def updateBandwidth(self, X):
        self.bw = [ae.updateBandwidth(x) for (ae,x) in zip(self.gMixAE, X)]
        return self.bw

    def kdeWrapper(self, data, size=None, bw=None, Orth=None):
        x = data['x']
        n = x.shape[0]
        if size is not None:
            size = self.trueBatchSize(data, size)
        else:
            size = n
        ix = np.random.choice(n, size, replace=True)
        x = x[ix, :]
        R = self.responsibility(x)
        #R = np.ones_like(R)
        compSizes = [np.sum(r) for r in R.T]
        if bw is None:
            bw = [bandwidth(cSize) for cSize in compSizes]
        elif np.isscalar(bw):
            bw = [bw for i in np.arange(self.nComps)]
        Enc = [self.normalized_encoding(x, r)[0] for r in R.T]
        IX = [bernoulli.rvs(r).astype('bool') for r in R.T]
        #pdb.set_trace()
        Enc = [enc[ix, :] for (enc, ix) in zip(Enc, IX)]
        if Orth is not None:
            Enc = [np.matmul(enc, orth) for (enc, orth) in zip(Enc, Orth)]
        KdeFncs = [[kde(enc[:, i:i+1], b) for i in np.arange(self.latent_dim)] for (enc,b) in zip(Enc, bw)]
        u = self.randomUniform()
        probs = [[kdeFnc(u) for kdeFnc in kdeFncs] for kdeFncs in KdeFncs]
        #pdb.set_trace()
        return u, probs, Enc

    def loss(self, data, R=None):
        x = data['x']
        if R is None:
            R = self.responsibility(x)
        compSizes = [np.sum(r) for r in R.T]
        bw = [bandwidth(cSize) for cSize in compSizes]
        lossRec = self.reconstructionLoss(x)
        Enc = [self.normalized_encoding(x, R[:, i])[0] for i in np.arange(self.nComps)]
        lossGaussian = sum([self.GaussianLoss(Enc[i], R[:, i], bw[i]) for i in np.arange(len(Enc))])
        loss = lossRec + lossGaussian
        return loss, lossRec, lossGaussian

    def normalized_encoding(self, x, r):
        e = self.encoding(x)
        mean = np.average(e, axis=0, weights=r.flatten())
        std = np.sqrt((np.average(e**2, axis=0, weights=r.flatten()) - mean**2))
        en = (e - mean)/std
        #pdb.set_trace()
        return en, mean, std

    def normalized_encoder(self, x, r):
        e = self.encoder(x)
        mean, var = tf.nn.weighted_moments(e, axes=0, frequency_weights=tf.reshape(r, (-1,1)))
        #pdb.set_trace()
        std = tf.math.sqrt(var)
        en = (e - mean) / std
        #pdb.set_trace()
        return en, mean, std

    def responsibility(self, x):
        for i in np.arange(20):
            try:
                #pdb.set_trace()
                R = self.rNet.predict(x)
                break
            except tf.errors.InvalidArgumentError:
                print(Exception)
        return R

    def gaussianResponsibilityTF(self, Enc, R):
        Dens = [tf.reduce_prod(tfp.distributions.Normal(loc=0, scale=1).prob(enc), axis=1, keepdims=True) for enc in Enc]
        props = [tf.reduce_mean(r) for r in tf.transpose(R)]
        #pdb.set_trace()
        gaussianR = [a*d for (a,d) in zip(props,Dens)]
        denom = tf.add_n(gaussianR)
        gaussianR = [r/denom for r in gaussianR]
        return gaussianR

    def getTrainableVariables(self):
        #vars = self.rNet.trainable_variables + super(GaussianMMixAE, self).getTrainableVariables()
        vars = list(set(list(chain(ae.getTrainableVariables() for ae in self.gMixAE))))
        return vars


    def copy(self):
        gMixAE = [ae.copy() for ae in self.gMixAE]
        encoder = gMixAE[0].encoder
        decoder = gMixAE[0].decoder
        for ae in self.gMixAE:
            ae.encoder = encoder
            ae.decoder = decoder
        kwargs = {'input_dim': self.input_dim, 'latent_dim': self.latent_dim, 'nComps': self.nComps,
                  'nMix': self.nMix, 'membership': self.membership,
                  'encoder': encoder, 'decoder': decoder, 'gMixAE': gMixAE}
        ae = type(self)(**kwargs)
        # ae.latent_dim = self.latent_dim
        # ae.input_dim = self.input_dim
        return ae