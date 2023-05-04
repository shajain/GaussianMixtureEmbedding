import tensorflow as tf
import numpy as np
from scipy.stats import bernoulli
from KDE.kde import bandwidth
from KDE.kde import kde
from AutoEncoder.AutoEncoder import AutoEncoderBasic
import tensorflow_probability as tfp
#import tensorflow_probability as tfp
from GaussianEmbedding.GaussianAE import GaussianAE
from NN.models import BasicMultiClass as RNet
import pdb


class GaussianMixAE(GaussianAE):
    def __init__(self, input_dim=1, latent_dim=1, nComps=2, **kwargs):
        super(GaussianMixAE, self).__init__(input_dim, latent_dim, **kwargs)
        self.nComps = nComps
        if 'rNet' in kwargs:
            self.rNet = kwargs['rNet']
        else:
            rNetModelArgs = AutoEncoderBasic.ModelDEF.copy()
            rNetModelArgs['output_dim'] = nComps
            # pdb.set_trace()
            self.rNet = RNet(**rNetModelArgs)
            self.rNet.build((None, input_dim))
        #pdb.set_trace()

    def NormsAndDistancesTF(self, enc):
        norm = tf.norm(enc, axis=-1, keepdims=True) ** 2
        inner = tf.matmul(enc, tf.transpose(enc))
        distSq = norm + tf.transpose(norm) - 2 * inner
        return norm, distSq

    def NormsAndDistances(self, enc):
        norm = np.linalg.norm(enc, axis=-1, keepdims=True) ** 2
        inner = np.matmul(enc, np.transpose(enc))
        distSq = norm + np.transpose(norm) - 2 * inner
        return norm, distSq


    def GaussianPosteriorLossTF(self, Enc, R):
        gaussR = self.gaussianResponsibilityTF(Enc, R)
        loss = tf.reduce_mean((R-gaussR)**2)
        #pdb.set_trace()
        return loss




    def GaussianLossTF(self, enc, r, bw):
        n = tf.reduce_sum(r)
        c = 1/(2*(n**2)*np.sqrt(np.pi))
        r = tf.reshape(r, (-1, 1))
        norm, distSq = self.NormsAndDistancesTF(enc)
        distSq = tf.reshape(distSq, -1)
        distSq = distSq/(4*(bw**2))
        norm = norm / (2 + 4 * (bw ** 2))
        rSq = tf.reshape(tf.matmul(r, tf.transpose(r)), -1)
        #pdb.set_trace()
        term1 = (1 / bw) * tf.reduce_sum(rSq * self.phiDTF(distSq))
        term2 = (n**2)/np.sqrt(1+bw**2)
        term3 = (2*n/np.sqrt(0.5+bw**2))*tf.reduce_sum(r * self.phiDTF(norm))
        loss = c*(term1 + term2 - term3)
        #pdb.set_trace()
        return loss

    def GaussianLoss(self, enc, r, bw):
        n = np.sum(r)
        c = 1/(2*(n**2)*np.sqrt(np.pi))
        r = np.reshape(r, (-1, 1))
        norm, distSq = self.NormsAndDistances(enc)
        distSq = np.reshape(distSq, -1)
        distSq = distSq / (4 * (bw ** 2))
        norm = norm / (2 + 4 * (bw ** 2))
        rSq = np.reshape(np.matmul(r, np.transpose(r)), -1)
        term1 = (1/bw)*np.sum(rSq * self.phiD(distSq))
        term2 = (n**2)/np.sqrt(1+bw**2)
        term3 = (2*n/np.sqrt(0.5+bw**2))*np.sum(r * self.phiD(norm))
        loss = c*(term1 + term2 - term3)
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
        R = self.rNet(x)
        #R = self.R
        Enc = [self.normalized_encoder(x, R[:, i])[0] for i in np.arange(self.nComps)]
        #encT = tf.matmul(enc, self.orth)
        #norm, distSq = self.NormsAndDistancesTF(encT)
        #pdb.set_trace()
        lossGaussian = tf.add_n([self.GaussianLossTF(Enc[i], R[:, i], self.bw[i]) for i in np.arange(len(Enc))])
        lossPosterior = self.GaussianPosteriorLossTF(Enc, R)
        #lossCorr = self.CorrelationLossTF(enc)
        #loss = lossGaussian + lossCorr
        #pdb.set_trace()
        loss = lossRec + lossGaussian + lossPosterior
        #loss = lossRec + lossGaussian
        #loss = lossGaussian
        return loss

    def gradients(self, data, batchSize):
        #pdb.set_trace()
        self.randomUniform()
        # batchSize = self.trueBatchSize(data['x'], batchSize)
        # x = data['x']
        # n = x.shape[0]
        # ix = np.random.choice(n, batchSize, replace=True)
        # x = x[ix, :]
        x = self.subSample(data['x'], batchSize)
        self.updateBandwidth(x)
        # R = self.responsibility(x)
        # #self.R = np.ones_like(R)
        # #pdb.set_trace()
        # compSizes = [np.sum(r) for r in R.T]
        # self.bw = [bandwidth(cSize) for cSize in compSizes]
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            # pdb.set_trace()
            tape.watch(self.getTrainableVariables())
            loss = self.lossTF(x)
        return loss, tape.gradient(loss, self.getTrainableVariables())


    def trueBatchSize(self, x, batchSize):
        n = x.shape[0]
        ix = np.random.choice(n, batchSize, replace=True)
        xx = x[ix, :]
        R = self.responsibility(xx)
        compProp = [np.mean(r) for r in R.T]
        trueBatchSize = np.floor(min(batchSize/min(compProp), n)).astype('int32')
        return trueBatchSize


    def subSample(self, x, batchSize):
        batchSize = self.trueBatchSize(x, batchSize)
        n = x.shape[0]
        ix = np.random.choice(n, batchSize, replace=True)
        x = x[ix, :]
        return x

    def updateBandwidth(self, x):
        R = self.responsibility(x)
        compSizes = [np.sum(r) for r in R.T]
        self.bw = [bandwidth(cSize) for cSize in compSizes]
        return self.bw

    def kdeWrapper(self, data, size=None, bw=None, Orth=None):
        x = data['x']
        n = x.shape[0]
        if size is not None:
            size = self.trueBatchSize(x, size)
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
    # def normalized_encoding(self, x, r):
    #     e = self.encoding(x)
    #     en = (e - np.mean(e, axis=0, keepdims=True))/np.std(e, axis=0)
    #     return en

    # def normalized_encoder(self, x, r):
    #     e = self.encoder(x)
    #     en = (e - tf.reduce_mean(e, axis=0, keepdims=True)) / tfp.stats.stddev(e, sample_axis=0)
    #     #pdb.set_trace()
    #     return en

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
        vars = self.rNet.trainable_variables + super(GaussianMixAE, self).getTrainableVariables()
        #vars = super(GaussianMixAE, self).getTrainableVariables()
        return vars
        #return self.encoder.trainable_variables + self.decoder.trainable_variables

    def copyRNet(self):
        rNet = self.rNet.copy()
        return rNet

    def copy(self):
        kwargs = {'input_dim': self.input_dim, 'latent_dim': self.latent_dim, 'nComps': self.nComps,
                  'encoder': self.copyEncoder(), 'decoder': self.copyDecoder(), 'rNet': self.copyRNet()}
        ae = type(self)(**kwargs)
        # ae.latent_dim = self.latent_dim
        # ae.input_dim = self.input_dim
        return ae