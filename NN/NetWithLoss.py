import tensorflow as tf
import numpy as np
from abc import ABC, abstractmethod
import pdb


class NetWithLoss(ABC):
    def __init__(self, net):
        self.net = net

    def getNet(self):
        return self.net

    def copyNet(self):
        return self.net.copy()

    def copy(self):
        return type(self)(self.copyNet())

    def setNet(self, net):
        self.net = net

    @abstractmethod
    def gradients(self, x, y, w, batchSize):
        pass

    @abstractmethod
    def loss(self, x, y, w):
        pass



