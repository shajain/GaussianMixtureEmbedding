#import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np
from GaussianLoss.specialFunctions import PhiD

class CWLoss:
    def __int__(self, dim):
        self.phiDTF = PhiD(dim, TF=True)
        self.phiD = PhiD(dim, TF=False)

    def GaussianLossTF(self, x, r, bw):
        n = tf.reduce_sum(r)
        c = 1/(2*(n**2)*np.sqrt(np.pi))
        r = tf.reshape(r, (-1, 1))
        norm, distSq = NormsAndDistancesTF(x)
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

    def GaussianLoss(self, x, r, bw):
        n = np.sum(r)
        c = 1/(2*(n**2)*np.sqrt(np.pi))
        r = np.reshape(r, (-1, 1))
        norm, distSq = NormsAndDistances(x)
        distSq = np.reshape(distSq, -1)
        distSq = distSq / (4 * (bw ** 2))
        norm = norm / (2 + 4 * (bw ** 2))
        rSq = np.reshape(np.matmul(r, np.transpose(r)), -1)
        term1 = (1/bw)*np.sum(rSq * self.phiD(distSq))
        term2 = (n**2)/np.sqrt(1+bw**2)
        term3 = (2*n/np.sqrt(0.5+bw**2))*np.sum(r * self.phiD(norm))
        loss = c*(term1 + term2 - term3)
        return loss


def NormsAndDistancesTF(self, x):
        norm = tf.norm(x, axis=-1, keepdims=True) ** 2
        inner = tf.matmul(x, tf.transpose(x))
        distSq = norm + tf.transpose(norm) - 2 * inner
        return norm, distSq

def NormsAndDistances(self, x):
        norm = np.linalg.norm(x, axis=-1, keepdims=True) ** 2
        inner = np.matmul(x, np.transpose(x))
        distSq = norm + np.transpose(norm) - 2 * inner
        return norm, distSq