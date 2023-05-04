
import pdb

import tensorflow as tf
import numpy as np
from KDE.kde import bandwidth
from KDE.kde import kde
from GaussianEmbedding.GaussianAE import GaussianAE
from GMM.classes import GMM
from scipy.stats import norm
import tensorflow_probability as tfp


class GaussianMixAE(GaussianAE):
    iterGMM = 10
    def __init__(self, input_dim=1, latent_dim=1, nComps = 2, **kwargs):
        super(GaussianMixAE, self).__init__(input_dim, latent_dim, **kwargs)
        self.bws = [bandwidth(self.batchSize)]*nComps
        self.nComps = nComps
        if 'GMM' in kwargs:
            self.GMM = kwargs['GMM']
        else:
            self.GMM = GMM(nComps, latent_dim, GaussianMixAE.iterGMM, spherical=True)

    # def GaussianLossTF(self, x, bw, mean, std):
    #     #n = x.shape[0]
    #     enc = self.encoder(x)
    #     enc_normalized = (enc - mean)/std
    #     kdeFnc = kde(enc_normalized, bw)[1]
    #     u = self.randomUniform()
    #     probs = kdeFnc(u)
    #     loss = tf.reduce_mean((probs - norm.pdf(self.u))**2)
    #     #pdb.set_trace()
    #     return loss

    def lossTF(self, cSamples):
        lossRec = tf.add_n([self.reconstructionLossTF(smp) for smp in cSamples])
        enc_ns = self.normalized_encodingsTF(cSamples)
        lossGaussian_c = [self.GaussianLossTF(enc_n, bw) for (enc_n, bw) in zip(enc_ns, self.bws)]
        lossGaussian = tf.add_n(lossGaussian_c)
        mean_enc_ns = [tf.reduce_mean(enc_n, axis=0) for enc_n in enc_ns]
        lossMean_c = [tf.nn.relu(tf.math.abs(mean)-0.25) for mean in mean_enc_ns]
        lossMean = tf.add_n(lossMean_c)
        #pdb.set_trace()
        return lossRec + lossGaussian + 100*lossMean


    def gradients(self, data, batchSize):
        x = data['x']
        n = x.shape[0]
        runGMM = False
        if not hasattr(self, 'R'):
            runGMM = True
        else:
            cSamples = self.componentSamples(self.x, self.R, batchSize)
            enc_ns = self.normalized_encodings(cSamples)
            mean_enc_ns = [np.mean(enc_n, axis=0) for enc_n in enc_ns]
            lossMean_c = np.array([np.abs(mean)-0.25 for mean in mean_enc_ns])
            lossMean_c[lossMean_c < 0] = 0
            lossMean = np.sum(lossMean_c)
            if lossMean == 0:
                runGMM = True
        if runGMM:
            self.x = x
            ix = np.random.choice(n, self.nComps*batchSize, replace=True)
            xx = self.x[ix, :]
            enc = self.encoding(xx)
            #pdb.set_trace()
            self.GMM.run(enc)
            self.updateGMMPars()
            self.R = self.responsibility_x(self.x)
        cSamples = self.componentSamples(self.x, self.R, batchSize)
        #pdb.set_trace()
        self.bws = [max(bandwidth(batchSize), bandwidth(sum(r))) for r in self.R.T]
        #self.nSamples = [min(batchSize, sum(r)) for r in self.R.T]
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            # pdb.set_trace()
            tape.watch(self.getTrainableVariables())
            loss = self.lossTF(cSamples)
        return loss, tape.gradient(loss, self.getTrainableVariables())

    def componentSamples(self, x, R, size=None):
        #R = self.responsibility_x(x)
        #pdb.set_trace()
        if size is None:
            compSamps = [x[np.random.choice(x.shape[0], np.sum(r).astype('int64'), replace=True, p=r/np.sum(r)), :] for r in R.T]
        else:
            compSamps = [x[np.random.choice(x.shape[0], size, replace=True, p=r/np.sum(r)), :] for r in R.T]
        return compSamps

    def normalized_encodingsTF(self, cSamples):
        encs_n = [(self.encoder(smp) - mean)/std for (smp, mean, std) in zip(cSamples, self.means, self.stds)]
        return encs_n

    def normalized_encodings(self, cSamples):
        encs_n = [(self.encoding(smp) - mean)/std for (smp, mean, std) in zip(cSamples, self.means, self.stds)]
        return encs_n


    def kdeWrapper(self, data, size=None):
        x = data['x']
        #pdb.set_trace()
        R = self.responsibility_x(x)
        if size is not None:
            cSamples = self.componentSamples(x, R, size)
            bws = [max(bandwidth(size), bandwidth(sum(r))) for r in R.T]
        else:
            cSamples = self.componentSamples(x, R)
            bws = [bandwidth(sum(r)) for r in R.T]
        kdeFncs = [kde(enc_n, bw)[0] for (enc_n, bw) in zip(self.normalized_encodings(cSamples), bws)]
        u = self.randomUniform()
        probs_c = [(1/std)*kdeFnc(u) for (kdeFnc, std) in zip(kdeFncs, self.stds)]
        u_c = [u*std + mean for (mean, std) in zip(self.means, self.stds)]
        u_m = np.array([uu for uu in u_c]).flatten()
        probs_m = sum([(1/std) * kdeFnc((u_m -mean)/std)*pi for (kdeFnc, mean, std, pi) in zip(kdeFncs, self.means, self.stds, self.pi)])
        return u_c, probs_c, u_m, probs_m

    def loss(self, data):
        x = data['x']
        size = 500
        R = self.responsibility_x(x)
        cSamples = self.componentSamples(x, R, size)
        bws = [max(bandwidth(size), bandwidth(sum(r))) for r in R.T]
        #pdb.set_trace()
        lossRec_c = [self.reconstructionLoss(smp) for smp in cSamples]
        lossGaussian_c = [self.GaussianLoss(enc_n, bw) for (enc_n, bw) in zip(self.normalized_encodings(cSamples), bws)]
        lossRec = sum(lossRec_c)
        lossGaussian = sum(lossGaussian_c)
        return lossRec + lossGaussian, lossRec_c, lossGaussian_c

    def updateGMMPars(self):
        self.means = [comp.mu for comp in self.GMM.model.comps]
        self.stds = [np.sqrt(np.diag(comp.cov)) for comp in self.GMM.model.comps]
        self.pi = self.GMM.model.pi

    def posterior(self, x):
        enc = self.encoding(x)
        return self.GMM.model.posterior(enc)


    def responsibility_x(self, x):
        enc = self.encoding(x)
        R_x = self.GMM.model.responsibility(enc)
        return R_x

    def copy(self):
        kwargs = dict()
        kwargs['encoder'] = self.copyEncoder()
        kwargs['decoder'] = self.copyDecoder()
        #pdb.set_trace()
        kwargs['GMM'] = self.GMM.copy()
        GMAE = GaussianMixAE(self.input_dim, self.latent_dim, self.nComps, **kwargs)
        GMAE.updateGMMPars()
        return GMAE

