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


class GaussianMultiMixAE(AutoEncoderBasic):

    def __init__(self, input_dim=1, latent_dim=1, nComps=2, nMix=2, cMemPerSample=None, **kwargs):
        # membership is a list of numpy vectors that tells which components are present in which sample.
        # The ith vector in the list contains the index of components present in that mixture.
        super(GaussianMultiMixAE, self).__init__(input_dim, latent_dim, **kwargs)
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

    def consistencyLossTF(self, X):
        #pdb.set_trace()
        MuVar = [[gae.normalized_encoder(x, r)[1:3] for r in tf.transpose(gae.rNet(x))] for (gae,x) in zip(self.gMixAE, X)]
        Mu_smp = [[mv[0] for mv in MV] for MV in MuVar]
        Var_smp = [[mv[1] for mv in MV] for MV in MuVar]
        Mu = [tf.stack([Mu_smp[i][j] for (i,j) in zip(sMem, cInd)]) for (sMem,cInd) in zip(self.sMemPerComp, self.cIndPerComp)]
        Var = [tf.stack([Var_smp[i][j] for (i,j) in zip(sMem, cInd)]) for (sMem,cInd) in zip(self.sMemPerComp, self.cIndPerComp)]
        loss = tf.reduce_sum(tf.add_n([tfp.stats.stddev(z, sample_axis=0) for z in Mu+Var]))
        return loss

    def consistencyLoss(self, X):
        MuVar = [[gae.normalized_encoding(x, r)[1:3] for r in gae.rNet(x).T] for (gae,x) in zip(self.gMixAE, X)]
        Mu_smp = [[mv[0] for mv in MV] for MV in MuVar]
        Var_smp = [[mv[1] for mv in MV] for MV in MuVar]
        Mu = [np.vstack([Mu_smp[i][j] for (i,j) in zip(sMem, cInd)]) for (sMem,cInd) in zip(self.sMemPerComp, self.cIndPerComp)]
        Var = [np.vstack([Var_smp[i][j] for (i,j) in zip(sMem, cInd)]) for (sMem,cInd) in zip(self.sMemPerComp, self.cIndPerComp)]
        muLosses = [np.reduce_sum(np.std(z, sample_axis=0)) for z in Mu]
        stdLosses = [np.reduce_sum(np.std(z, sample_axis=0)) for z in Var]
        loss = sum(muLosses+stdLosses)
        return loss, muLosses, stdLosses

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


    # def trueBatchSize(self, data, batchSize):
    #     x = data['x']
    #     n = x.shape[0]
    #     ix = np.random.choice(n, batchSize, replace=True)
    #     xx = x[ix, :]
    #     R = self.responsibility(xx)
    #     compProp = [np.mean(r) for r in R.T]
    #     trueBatchSize = np.floor(min(batchSize/min(compProp), n)).astype('int32')
    #     return trueBatchSize

    # def subSample(self, X, batchSize):
    #     X = [ae.subSample(x, batchSize) for (ae, x) in zip(self.gMixAE, X)]
    #     return X
    #
    # def updateBandwidth(self, X):
    #     self.bw = [ae.updateBandwidth(x) for (ae,x) in zip(self.gMixAE, X)]
    #     return self.bw

    # def kdeWrapper(self, data, size=None, bw=None, Orth=None):
    #     x = data['x']
    #     n = x.shape[0]
    #     if size is not None:
    #         size = self.trueBatchSize(data, size)
    #     else:
    #         size = n
    #     ix = np.random.choice(n, size, replace=True)
    #     x = x[ix, :]
    #     R = self.responsibility(x)
    #     #R = np.ones_like(R)
    #     compSizes = [np.sum(r) for r in R.T]
    #     if bw is None:
    #         bw = [bandwidth(cSize) for cSize in compSizes]
    #     elif np.isscalar(bw):
    #         bw = [bw for i in np.arange(self.nComps)]
    #     Enc = [self.normalized_encoding(x, r)[0] for r in R.T]
    #     IX = [bernoulli.rvs(r).astype('bool') for r in R.T]
    #     #pdb.set_trace()
    #     Enc = [enc[ix, :] for (enc, ix) in zip(Enc, IX)]
    #     if Orth is not None:
    #         Enc = [np.matmul(enc, orth) for (enc, orth) in zip(Enc, Orth)]
    #     KdeFncs = [[kde(enc[:, i:i+1], b) for i in np.arange(self.latent_dim)] for (enc,b) in zip(Enc, bw)]
    #     u = self.randomUniform()
    #     probs = [[kdeFnc(u) for kdeFnc in kdeFncs] for kdeFncs in KdeFncs]
    #     #pdb.set_trace()
    #     return u, probs, Enc

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

    # def normalized_encoding(self, x, r):
    #     e = self.encoding(x)
    #     mean = np.average(e, axis=0, weights=r.flatten())
    #     std = np.sqrt((np.average(e**2, axis=0, weights=r.flatten()) - mean**2))
    #     en = (e - mean)/std
    #     #pdb.set_trace()
    #     return en, mean, std
    #
    # def normalized_encoder(self, x, r):
    #     e = self.encoder(x)
    #     mean, var = tf.nn.weighted_moments(e, axes=0, frequency_weights=tf.reshape(r, (-1,1)))
    #     #pdb.set_trace()
    #     std = tf.math.sqrt(var)
    #     en = (e - mean) / std
    #     #pdb.set_trace()
    #     return en, mean, std

    # def responsibility(self, x):
    #     for i in np.arange(20):
    #         try:
    #             #pdb.set_trace()
    #             R = self.rNet.predict(x)
    #             break
    #         except tf.errors.InvalidArgumentError:
    #             print(Exception)
    #     return R

    # def gaussianResponsibilityTF(self, Enc, R):
    #     Dens = [tf.reduce_prod(tfp.distributions.Normal(loc=0, scale=1).prob(enc), axis=1, keepdims=True) for enc in Enc]
    #     props = [tf.reduce_mean(r) for r in tf.transpose(R)]
    #     #pdb.set_trace()
    #     gaussianR = [a*d for (a,d) in zip(props,Dens)]
    #     denom = tf.add_n(gaussianR)
    #     gaussianR = [r/denom for r in gaussianR]
    #     return gaussianR

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