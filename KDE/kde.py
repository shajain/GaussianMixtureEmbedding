import pdb

import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from scipy.stats import norm
from scipy.stats import uniform
from scipy.special import logsumexp


def bandwidth(n):
    bw = 1.06 * (n ** (-0.2))
    return bw

def kde(x,bw=None):
    n = x.shape[0]
    if bw == None:
        bw = bandwidth(n)
    if isinstance(x, np.ndarray):
        bw = bw * np.std(x.flatten())
        def kdeFnc(xx):
            distSq = (xx - x) ** 2
            exponent = -distSq / (2 * (bw ** 2))
            exps = np.exp(exponent)
            allprobs = ((np.sqrt(2 * np.pi) * bw) ** (-1)) * exps
            probs = np.mean(allprobs, axis=0)
            return probs
    elif tf.is_tensor(x):
        bw = bw * tf.math.reduce_std(tf.squeeze(x))
        def kdeFnc(xx):
            distSq = (xx - x) ** 2
            #pdb.set_trace()
            exponent = -distSq / (2 * (bw ** 2))
            exps = tf.math.exp(exponent)
            allprobs = ((np.sqrt(2 * np.pi) * bw) ** (-1)) * exps
            probs = tf.reduce_mean(allprobs, axis=0)
            return probs
    return kdeFnc