import itertools

import tensorflow as tf
import numpy as np
from scipy.stats import norm
from scipy.stats import uniform
from scipy.special import logsumexp
from KDE.kde import bandwidth
from KDE.kde import kde
from AutoEncoder.AutoEncoder import AutoEncoderBasic
import tensorflow_probability as tfp
#import tensorflow_probability as tfp
from matplotlib import pyplot as plt
from misc import sortedplot as sp
from GaussianLoss.specialFunctions import PhiD
import pdb


class GaussianAE(AutoEncoderBasic):
    def __init__(self, input_dim=1, latent_dim=1, **kwargs):
        super(GaussianAE, self).__init__(input_dim, latent_dim, **kwargs)
        self.bw = bandwidth(self.batchSize)
        self.nRand = 100
        self.randomGaussian()
        self.randomUniform()
        self.lDimPairs = list(itertools.combinations(np.arange(latent_dim), 2))
        self.orth = np.identity(latent_dim)
        self.phiDTF = PhiD(latent_dim, TF=True)
        self.phiD = PhiD(latent_dim, TF=False)
        #pdb.set_trace()

    def randomUniformAndGaussian(self):
        u = self.randomUniform()
        g = self.randomGaussian()
        self.ug = np.concatenate((u, g), axis=0)
        return self.ug

    def randomGaussian(self):
        self.g = norm.rvs(size=self.nRand)
        return self.g
    def randomUniform(self):
        self.u = uniform(loc=-4, scale=8).rvs(size=self.nRand)
        return self.u

    def GaussianLossTF2(self, enc, bw):
        n = enc.shape[0]
        #enc = self.normalized_encoder(x)
        # std = tfp.stats.stddev(enc)
        # mean = tf.reduce_mean(enc, axis=0, keepdims=True)
        # enc_normalized = (enc - mean)/std
        distSq = (self.g - enc)**2
        exponent = -distSq/(2*(bw**2))
        logprobs = tf.reduce_logsumexp(exponent, axis=0)
        loss =  tf.math.log(n*tf.math.sqrt(2*np.pi)*self.bw) - tf.reduce_mean(logprobs)
        #pdb.set_trace()
        return loss


    def CorrelationLossTF(self, enc):
        corr = tfp.stats.correlation(enc, sample_axis=0)
        # apply mask to x and sum non-diagonal entries
        mask = tf.ones_like(corr) - tf.eye(tf.shape(corr)[0])
        abs_corr = tf.math.abs(tf.reduce_sum(tf.multiply(corr, mask)))
        loss = 10*tf.nn.relu(abs_corr - 0.01)
        return loss

    def GaussianLossTF(self, enc, bw):
        n = enc.shape[0]
        #pdb.set_trace()
        c = 1/(2*(n**2)*np.sqrt(np.pi))
        norm = tf.norm(enc, axis=-1, keepdims=True) ** 2
        inner = tf.matmul(enc, tf.transpose(enc))
        #norm1 = tf.norm(enc[:, None, :] - enc, axis=-1)/(4*(bw**2))
        norm1 = (norm + tf.transpose(norm) - 2 * inner)/(4*(bw**2))
        norm1 = tf.reshape(norm1, -1)
        norm2 = norm / (2 + 4 * (bw ** 2))
        #pdb.set_trace()
        #term1 = (1/bw)*tf.reduce_sum(tf.map_fn(self.phiDTF, norm1))
        term1 = (1 / bw) * tf.reduce_sum(self.phiDTF(norm1))
        #pdb.set_trace()
        term2 = (n**2)/np.sqrt(1+bw**2)
        term3 = (2*n/np.sqrt(0.5+bw**2))*tf.reduce_sum(self.phiDTF(norm2))
        loss = c*(term1 + term2 - term3)
        #pdb.set_trace()
        return loss

    def GaussianLoss(self, enc, bw):
        n = enc.shape[0]
        c = 1/(2*(n**2)*np.sqrt(np.pi))
        norm = np.linalg.norm(enc, axis=-1, keepdims=True) ** 2
        inner = np.matmul(enc, np.transpose(enc))
        # norm1 = tf.norm(enc[:, None, :] - enc, axis=-1)/(4*(bw**2))
        norm1 = (norm + np.transpose(norm) - 2 * inner) / (4 * (bw ** 2))
        norm1 = np.reshape(norm1, -1)
        norm2 = norm / (2 + 4 * (bw ** 2))
        #norm1 = (np.linalg.norm(enc[:, None, :] - enc, axis=-1)**2)/(4*(bw**2))
        #norm2 = np.linalg.norm(enc, axis=-1) / (2 + 4 * (bw ** 2))
        #pdb.set_trace()
        term1 = (1/bw)*np.sum(self.phiD(norm1))
        term2 = (n**2)/np.sqrt(1+bw**2)
        term3 = (2*n/np.sqrt(0.5+bw**2))*np.sum(self.phiD(norm2))
        loss = c*(term1 + term2 - term3)
        return loss

    def GaussianLossTF1(self, enc, bw):
        #n = x.shape[0]
        #enc = self.encoder(x)
        # std = tfp.stats.stddev(enc)
        # mean = tf.reduce_mean(enc, axis=0, keepdims=True)
        #enc_normalized = (enc - mean)/std
        kdeFnc = kde(enc, bw)
        # distSq = (self.u - enc_normalized)**2
        # exponent = -distSq/(2*(self.bw**2))
        # exps = tf.math.exp(exponent)
        # allprobs = ((np.sqrt(2*np.pi) * self.bw)**(-1))*exps
        # probs = tf.reduce_mean(allprobs, axis = 0)
        u = self.randomUniform()
        g = self.randomGaussian()
        e = enc.numpy().flatten()
        v = np.hstack((u,g,e))
        probs = kdeFnc(v)
        loss = tf.reduce_mean((probs - norm.pdf(v))**2)
        sp.sortedplot(v, probs.numpy())
        sp.sortedplot(v, norm.pdf(v))
        sp.title('tensorflow')
        sp.show()
        #pdb.set_trace()
        return loss

    def testGradient(self, data, batchSize):
        self.randomUniform()
        self.bw = bandwidth(batchSize)
        x = data['x']
        n = x.shape[0]
        ix = np.random.choice(n, batchSize, replace=True)
        x = x[ix, :]
        enc = self.normalized_encoding(x)
        encTF = tf.constant(enc)
        xx = tf.constant([[0.0,8.0, 9.0], [0.0, 5.0, 6.0]])
        #xx = tf.constant([0.0])
        #norm1 = tf.constant( [[3.0517578e-05, 2.7000000e+01], [2.7000000e+01, 0.0000000e+00]])
        #x = tf.Variable(3.0)
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            # pdb.set_trace()
            tape.watch(xx)
            #norm2 = tf.norm(enc, axis=-1) / (2 + 4 * (self.bw ** 2))
            #norm1 = tf.norm(enc[:, None, :] - enc, axis=-1) / (4 * (self.bw ** 2))
            #norm1 = tf.norm(encTF[:, None, :] -encTF, axis=-1)
            # norm = tf.norm(xx, axis=-1, keepdims=True) ** 2
            # inner = tf.matmul(xx, tf.transpose(xx))
            # norm1 = norm + tf.transpose(norm) - 2 * inner
            # norm = tf.norm(x[:, None, :]-x, axis=-1)
            # norm1 = tf.reshape(norm1, -1)
            # loss = tf.reduce_sum(self.phiDTF(norm1))
            loss = tf.reduce_sum(self.phiDTF(xx))
            #loss = tf.where(xx<5.0, xx, xx)
        g = tape.gradient(loss, xx)

        pdb.set_trace()
        return

    def lossTF(self, x):
        lossRec = self.reconstructionLossTF(x)
        enc = self.normalized_encoder(x)
        encT = tf.matmul(enc, self.orth)
        #pdb.set_trace()
        #lossGaussian = tf.add_n([self.GaussianLossTF(encT[:, i:i+1], self.bw) for i in np.arange(self.latent_dim)])
        lossGaussian = self.GaussianLossTF(encT, self.bw)
        #lossCorr = self.CorrelationLossTF(enc)
        #loss = lossGaussian + lossCorr
        #pdb.set_trace()
        loss = lossRec + lossGaussian
        #loss = lossGaussian
        return loss

    def gradients(self, data, batchSize):
        #pdb.set_trace()
        self.randomUniform()
        self.bw = bandwidth(batchSize)
        x = data['x']
        n = x.shape[0]
        ix = np.random.choice(n, batchSize, replace=True)
        x = x[ix, :]
        enc = self.normalized_encoding(x)
        # if np.random.uniform(size=1) < 0.0:
        #     self.orth = pca(enc)
        # else:
        #     self.orth = ortho_group.rvs(self.latent_dim)
        e = np.matmul(enc, self.orth)
        plt.show()
        plt.scatter(e[:,0], e[:,1])
        plt.show()
        #pdb.set_trace()
        u, probs = self.kdeWrapper(data, 500, bw=None, orth = self.orth)
        sp.sortedplot(u,probs[0])
        sp.sortedplot(u, norm.pdf(u))
        sp.show()
        sp.sortedplot(u, probs[1])
        sp.sortedplot(u, norm.pdf(u))
        sp.show()
        #pdb.set_trace()
        return super().gradients(data, batchSize)

    def GaussianLoss1(self, enc, bw=None):
        # x = data['x']
        # n = x.shape[0]
        # ix = np.random.choice(n, self.batchSize, replace=True)
        # x = x[ix, :]
        # enc = self.normalized_encoding(x)
        # std = np.std(enc)
        # mean = np.mean(enc, axis=0, keepdims=True)
        # enc_normalized = (enc - mean) / std
        dist = (self.g - enc) ** 2
        exponent = -dist / (self.bw ** 2)
        logprobs = logsumexp(exponent, axis=0)
        loss = - np.mean(logprobs)
        return loss

    def GaussianLoss2(self, enc, bw=None):
        #enc = data['enc']
        #enc = self.encoding(x)
        # std = np.std(enc)
        # mean = np.mean(enc, axis=0, keepdims=True)
        # enc_normalized = (enc - mean) / std
        kdeFnc = kde(enc, bw)
        # distSq = (self.u - enc_normalized) ** 2
        # exponent = -distSq / (2 * (self.bw ** 2))
        # exps = np.exp(exponent)
        # allprobs = ((np.sqrt(2 * np.pi) * self.bw) ** (-1)) * exps
        # probs = np.mean(allprobs, axis=0)
        u = self.randomUniform()
        probs = kdeFnc(u)
        loss = np.mean((probs - norm.pdf(self.u)) ** 2)
        # pdb.set_trace()
        return loss

    def kdeWrapper(self, data, size=None, bw=None, orth=None):
        x = data['x']
        n = x.shape[0]
        if size is not None:
            ix = np.random.choice(n, size, replace=True)
            x = x[ix, :]
            n = size
        if bw is None:
            bw = bandwidth(n)
        enc = self.normalized_encoding(x)
        if orth is not None:
            enc = np.matmul(enc, orth)
        kdeFncs = [kde(enc[:, i:i+1], bw) for i in np.arange(self.latent_dim)]
        u = self.randomUniform()
        probs = [kdeFnc(u) for kdeFnc in kdeFncs]
        #pdb.set_trace()
        return u, probs

    def loss(self, data):
        x = data['x']
        n = x.shape[0]
        bw = bandwidth(n)
        lossRec = self.reconstructionLoss(x)
        enc = self.normalized_encoding(x)
        lossGaussian = sum([self.GaussianLoss(enc[:, i:i+1], bw) for i in np.arange(self.latent_dim)])
        loss = lossRec + lossGaussian
        return loss, lossRec, lossGaussian

    def normalized_encoding(self, x):
        e = self.encoding(x)
        en = (e - np.mean(e, axis=0, keepdims=True))/np.std(e, axis=0)
        return en

    def normalized_encoder(self, x):
        e = self.encoder(x)
        en = (e - tf.reduce_mean(e, axis=0, keepdims=True)) / tfp.stats.stddev(e, sample_axis=0)
        #pdb.set_trace()
        return en