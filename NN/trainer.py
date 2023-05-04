import tensorflow as tf
import numpy as np
from abc import ABC, abstractmethod
import pdb

class Trainer:
    def __init__(self, netWithLoss, data, maxIter, batchSize):
        # pdb.set_trace()
        self.opt = tf.keras.optimizers.Adam( )
        self.nnLoss = netWithLoss
        self.data = data
        self.batchSize = batchSize
        self.maxIter = maxIter
        self.loss = []

    def loss(self):
        return self.nnLoss.loss(self.data)

    def fit(self):
        #self.beforeTraining()
        #pdb.set_trace()
        for i in np.arange(self.maxIter):
            self.iter = i
            self.iteration( )
        #pdb.set_trace()
        return

    def iteration(self):
        #pdb.set_trace()
        self.beforeUpdate()
        #pdb.set_trace()
        loss, gradients = self.nnLoss.gradients(self.data, self.batchSize)
        #pdb.set_trace()
        #self.nnLoss.testGradient(self.data,self.batchSize)
        self.opt.apply_gradients(zip(gradients, self.nnLoss.getTrainableVariables()))
        self.loss.append(loss)
        self.afterUpdate()
        return

    def attachDebugger(self, debug):
        self.debug = debug
        self.nets = []
        self.debug.attachNets(self.nets)

    def beforeTraining(self):
        if hasattr(self, 'debug'):
            self.nets.append(self.nnLoss.copy( ))

    def beforeUpdate(self):
        if hasattr(self, 'debug'):
            self.debug.beforeUpdate(self.iter)

    def afterUpdate(self):
        if hasattr(self, 'debug'):
            self.nets.append(self.nnLoss.copy( ))
            #pdb.set_trace()
            self.debug.afterUpdate(self.loss[-1])



class AlternatingTrainer:
    def __init__(self, muNet, sigmaNet, x, y, w, rounds, maxIter, batchSize):
        #pdb.set_trace()
        self.rounds = rounds
        self.maxIter = maxIter
        self.batchSize = batchSize
        self.muTrainer = Trainer(muNet, x, y, w, maxIter, batchSize)
        self.sigmaTrainer = Trainer(sigmaNet, x, y, w, maxIter, batchSize)
        self.muNet = muNet
        self.sigmaNet = sigmaNet
        self.muNet.attachSigmaNet(sigmaNet.net)
        self.sigmaNet.attachMuNet(muNet.net)
        #pdb.set_trace()
        #self.debug.attachData(x,y)

    def attachDebugger(self, debug):
        self.debug = debug
        self.muNets = []
        self.sigmaNets = []
        self.debug.attachMuNets(self.muNets)
        self.debug.attachSigmaNets(self.sigmaNets)

    def beforeMuUpdate(self):
        if hasattr(self, 'debug'):
            self.muNets.append(self.muNet.copy())
            self.debug.beforeMuUpdate(self.round)

    def beforeSigmaUpdate(self):
        if hasattr(self, 'debug'):
            self.sigmaNets.append(self.sigmaNet.copy())
            self.debug.beforeSigmaUpdate()

    def endOfRound(self):
        if hasattr(self, 'debug'):
            self.debug.endOfRound()



    def fit(self):
        for i in np.arange(self.rounds):
            self.round = i
            self.iteration()
        return

    def iteration(self):
        #pdb.set_trace()
        self.beforeMuUpdate()
        self.muTrainer.fit( )

        self.beforeSigmaUpdate()
        self.sigmaTrainer.fit( )

        self.endOfRound()
        return