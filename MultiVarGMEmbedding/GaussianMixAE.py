import pdb

import tensorflow as tf
import numpy as np
from scipy.stats import pearsonr
from KDE.kde import bandwidth
from KDE.kde import kde
from GaussianEmbedding.GaussianAE import GaussianAE
from GMM.classes import GMM
from itertools import combinations
from scipy.stats import norm
from PCA.pca import pca
import tensorflow_probability as tfp
from scipy.stats import ortho_group


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
        encT_ns = [tf.matmul(enc_n-tf.reduce_mean(enc_n, axis=0), orth) for (enc_n, orth) in zip(enc_ns, self.orths)]
        #pdb.set_trace()
        lossGaussian_c = [[self.GaussianLossTF(enc_n[:, i:i+1], bw) for i in np.arange(enc_n.shape[1])]for (enc_n, bw) in zip(encT_ns, self.bws)]
        lossGaussian = tf.add_n([tf.add_n(lossGaussian_c[i]) for i in np.arange(len(lossGaussian_c))])
        mean_enc_ns = [tf.reduce_mean(enc_n, axis=0) for enc_n in enc_ns]
        lossMean_c = [tf.nn.relu(tf.math.abs(mean)-0.25) for mean in mean_enc_ns]
        lossMean = tf.add_n(lossMean_c)
        corr_c = [[tf.math.abs(tfp.stats.correlation(enc_n[:, i:i+1], enc_n[:, j:j+1], sample_axis=0))
                   for (i, j) in combinations(np.arange(self.latent_dim), 2)] for enc_n in encT_ns]
        lossCorr = tf.add_n([tf.add_n([tf.nn.relu(corr-0.01) for corr in corrs]) for corrs in corr_c])
        #pdb.set_trace()
        return lossRec + 10*lossGaussian + 0*lossMean + 0*lossCorr


    def gradients(self, data, batchSize):
        x = data['x']
        n = x.shape[0]

        # runGMM = False
        # if not hasattr(self, 'R'):
        #     runGMM = True
        # else:
        #     cSamples = self.componentSamples(self.x, self.R, batchSize)
        #     enc_ns = self.normalized_encodings(cSamples)
        #     mean_enc_ns = [np.mean(enc_n, axis=0) for enc_n in enc_ns]
        #     lossMean_c = np.array([np.abs(mean)-0.25 for mean in mean_enc_ns])
        #     lossMean_c[lossMean_c < 0] = 0
        #     lossMean = np.sum(lossMean_c)
        #     if lossMean == 0:
        #         runGMM = True
        # if runGMM:
        #     self.x = x
        #     ix = np.random.choice(n, self.nComps*batchSize, replace=True)
        #     xx = self.x[ix, :]
        #     enc = self.encoding(xx)
        #     #pdb.set_trace()
        #     self.GMM.run(enc)
        #     self.updateGMMPars()
        #     self.R = self.responsibility_x(self.x)
        # Remove the six lines below for the actual algorithm
        self.x = x
        self.R = self.responsibility_x(self.x)
        enc = self.encoding(self.x)
        self.mean_enc = np.mean(enc, axis=0)
        self.std_enc = np.std(enc, axis=0)
        self.updateGMMPars()
        cSamples = self.componentSamples(self.x, self.R, batchSize)
        #pdb.set_trace()
        self.bws = [max(bandwidth(batchSize), bandwidth(sum(r))) for r in self.R.T]
        if np.random.uniform(size=1) > 0.5:
            self.orths = [pca(enc_n) for enc_n in self.normalized_encodings(cSamples)]
        else:
            self.orths = [ortho_group.rvs(self.latent_dim) for i in np.arange(len(cSamples))]
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
        # means = self.means
        # stds = self.stds
        encs = [self.encoder(csmp) for csmp in cSamples]
        means = [tf.reduce_mean(enc, axis=0) for enc in encs]
        stds = [tfp.stats.stddev(enc, sample_axis=0) for enc in encs]
        encs_n = [(enc - mean) / std for (enc, mean, std) in zip(encs, means, stds)]
        #encs_n = [(self.encoder(smp) - mean)/std for (smp, mean, std) in zip(cSamples, self.means, self.stds)]
        return encs_n

    def normalized_encodings(self, cSamples):
        # means =self.means
        # stds = self.stds
        encs = [self.encoding(csmp) for csmp in cSamples]
        means = [np.mean(enc, axis =0) for enc in encs]
        stds = [np.std(enc, axis=0) for enc in encs]
        encs_n = [(enc - mean)/std for (enc, mean, std) in zip(encs, means, stds)]
        return encs_n


    def kdeWrapper(self, data, size=None, orth=None):
        #Most likely when orth is not eqaul to identity this won't work
        if orth is None:
            orth = np.diag(np.ones(self.latent_dim))
        x = data['x']
        R = self.responsibility_x(x)
        if size is not None:
            cSamples = self.componentSamples(x, R, size)
            bws = [max(bandwidth(size), bandwidth(sum(r))) for r in R.T]
        else:
            cSamples = self.componentSamples(x, R)
            bws = [bandwidth(sum(r)) for r in R.T]
        enc_ns = self.normalized_encodings(cSamples)
        enc_ns = [np.matmul(enc, orth) for enc in enc_ns]
        means = self.means
        #means = [np.squeeze(np.matmul(mean[None,:], orth)) for mean in self.means]
        stds = [np.sqrt(np.diag(np.matmul(np.matmul(orth.T, np.diag(std**2)), orth))) for std in self.stds]
        #pdb.set_trace()
        kdeFncs = [[kde(enc_n[:, d:d+1], bw)[0] for d in np.arange(self.latent_dim)] for (enc_n, bw) in zip(enc_ns, bws)]
        u = self.randomUniform()
        #pdb.set_trace()
        probs_c_dim = [np.hstack([(1/std[i])*kdeFnc[i](u)[:, None] for i in np.arange(self.latent_dim)]) for (kdeFnc, std) in zip(kdeFncs, stds)]
        probs_c = [np.prod(probs, axis=1) for probs in probs_c_dim]
        u_c = [u[:, None]*std + mean for (mean, std) in zip(means, stds)]
        u_m = np.vstack([uu for uu in u_c])
        probs_m_dim = [np.hstack([(1 / std[i]) * kdeFnc[i]((u_m[:, i]-mean[i])/std[i])[:, None] for i in np.arange(self.latent_dim)]) for
                       (kdeFnc, mean, std) in zip(kdeFncs, means, stds)]
        probs_m = sum([np.prod(probs, axis=1)*pi for (probs, pi) in zip(probs_m_dim, self.pi)])
        probs_m_dim = sum([probs*pi for (probs, pi) in zip(probs_m_dim, self.pi)])
        return u_c, probs_c_dim, probs_c, u_m, probs_m_dim, probs_m

    def loss(self, data):
        x = data['x']
        size = 500
        R = self.responsibility_x(x)
        cSamples = self.componentSamples(x, R, size)
        bws = [max(bandwidth(size), bandwidth(sum(r))) for r in R.T]
        #pdb.set_trace()
        lossRec_c = [self.reconstructionLoss(smp) for smp in cSamples]
        lossRec = sum(lossRec_c)

        enc_ns = self.normalized_encodings(cSamples)
        encT_ns = [np.matmul(enc_n-np.mean(enc_n, axis=0), orth) for (enc_n, orth) in zip(enc_ns, self.orths)]
        lossGaussian_c = [sum([self.GaussianLoss(enc_n[:, i:i + 1], bw) for i in np.arange(self.latent_dim)]) for (enc_n, bw) in zip(encT_ns, self.bws)]
        lossGaussian = sum(lossGaussian_c)

        corr_c = [[np.abs(pearsonr(enc_n[:, i], enc_n[:, j])[0]) for (i, j) in
                   combinations(np.arange(enc_n.shape[1]), 2)] for enc_n in encT_ns]
        lossCorr_c = [np.array([corr - 0.01 for corr in corrs]) for corrs in corr_c]
        for lc in lossCorr_c:
            lc[lc < 0] = 0
        lossCorr_c = [np.sum(lc) for lc in lossCorr_c]
        lossCorr = sum(lossCorr_c)
        #xpdb.set_trace()
        return lossRec + lossGaussian + lossCorr, lossRec_c, lossGaussian_c, lossCorr_c

    def updateGMMPars(self):
        # self.means = [comp.mu for comp in self.GMM.model.comps]
        # self.stds = [np.sqrt(np.diag(comp.cov)) for comp in self.GMM.model.comps]
        # self.pi = self.GMM.model.pi
        #self.means = [np.zeros(self.latent_dim)]
        #self.stds = [np.ones(self.latent_dim)]
        self.means = [self.mean_enc]
        self.stds = [self.std_enc]
        self.pi = np.ones(1)

    def posterior(self, x):
        # enc = self.encoding(x)
        # p = self.GMM.model.posterior(enc)
        p = np.ones(x.shape[0])
        return p

    def responsibility_x(self, x):
        #enc = self.encoding(x)
        #R_x = self.GMM.model.responsibility(enc)
        R_x = np.ones((x.shape[0], 1))
        return R_x

    def copy(self):
        kwargs = dict()
        kwargs['encoder'] = self.copyEncoder()
        kwargs['decoder'] = self.copyDecoder()
        #pdb.set_trace()
        kwargs['GMM'] = self.GMM.copy()
        GMAE = GaussianMixAE(self.input_dim, self.latent_dim, self.nComps, **kwargs)
        GMAE.mean_enc = np.copy(self.mean_enc)
        GMAE.std_enc = np.copy(self.std_enc)
        GMAE.updateGMMPars()
        GMAE.orths = [np.copy(orth) for orth in self.orths]
        return GMAE

