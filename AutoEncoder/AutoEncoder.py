from NN.models import BasicRelu as EncoderModel
import tensorflow as tf
import numpy as np
from abc import ABC, abstractmethod
import pdb

class AbstractAutoEncoderWithLoss(ABC):
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder
        self.batchSize = 500

    def getEncoder(self):
        return self.encoder

    def getDecoder(self):
        return self.decoder

    def copyEncoder(self):
        return self.encoder.copy()

    def copyDecoder(self):
        return self.decoder.copy()

    def copy(self):
        kwargs = {'encoder':self.copyEncoder(), 'decoder': self.copyDecoder()}
        return type(self)(**kwargs)

    def setEncoder(self, encoder):
        self.encoder = encoder

    def setDecoder(self, decoder):
        self.decoder = decoder


    def decoding(self, e):
        for i in np.arange(20):
            try:
                xx = self.decoder.predict(e)
                break
            except tf.errors.InvalidArgumentError:
                print(Exception)
        return xx

    def encoding(self, x):
        for i in np.arange(20):
            try:
                e = self.encoder.predict(x)
                break
            except tf.errors.InvalidArgumentError:
                print(Exception)
        return e

    def reconstruction(self, x):
        e = self.encoding(x)
        return self.decoding(e)

    def encoding2reconstruction(self, e):
        return self.decoder.predict(e)

    def getTrainableVariables(self):
        return self.encoder.trainable_variables
        #return self.encoder.trainable_variables + self.decoder.trainable_variables

    @abstractmethod
    def gradients(self, data, batchSize):
        pass

    @abstractmethod
    def loss(self, data):
        pass


class AutoEncoderBasic(AbstractAutoEncoderWithLoss):
    ModelDEF = {'n_units': 20, 'n_hidden': 10, 'dropout_rate': 0.2}
    def __init__(self, input_dim=1, latent_dim=1, **kwargs):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        if 'encoder' in kwargs:
            encoder = kwargs['encoder']
        else:
            encModelArgs = AutoEncoderBasic.ModelDEF.copy()
            encModelArgs['output_dim'] = latent_dim
            #pdb.set_trace()
            encoder = EncoderModel(**encModelArgs)
            encoder.build((None, input_dim))
        if 'decoder' in kwargs:
            decoder = kwargs['decoder']
        else:
            decModelArgs = AutoEncoderBasic.ModelDEF.copy()
            decModelArgs['output_dim'] = input_dim
            decoder = EncoderModel(**decModelArgs)
            decoder.build((None, latent_dim))
        super(AutoEncoderBasic, self).__init__(encoder, decoder)

    def reconstructionLossTF(self, x):
        encoding = self.encoder(x)
        xx = self.decoder(encoding)
        loss = tf.reduce_mean(tf.reduce_sum((x - xx) ** 2, axis=1))
        #pdb.set_trace()
        return loss

    def lossTF(self, x):
        loss = self.reconstructionLossTF(x)
        return loss

    def gradients(self, data, batchSize):
        self.batchSize = batchSize
        x = data['x']
        n = x.shape[0]
        ix = np.random.choice(n, batchSize, replace=True)
        x = x[ix, :]
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            # pdb.set_trace()
            tape.watch(self.getTrainableVariables())
            loss = self.lossTF(x)
        return loss, tape.gradient(loss, self.getTrainableVariables())

    def reconstructionLoss(self, x):
        xx = self.reconstruction(x)
        loss = np.mean(np.sum((x - xx) ** 2, axis=1))
        #pdb.set_trace()
        return loss

    def loss(self, data):
        loss = self.reconstructionLoss(data['x'])
        return loss

    def copy(self):
        kwargs = {'input_dim': self.input_dim, 'latent_dim': self.latent_dim,
                  'encoder': self.copyEncoder(), 'decoder': self.copyDecoder()}
        ae = type(self)(**kwargs)
        # ae.latent_dim = self.latent_dim
        # ae.input_dim = self.input_dim
        return ae
