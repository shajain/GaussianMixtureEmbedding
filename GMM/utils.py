import pdb

import numpy as np
from scipy.special import logsumexp

def responsibilityAndLogp(x, comps, pi):
    logpdfs_c = np.array([comp.logpdf(x) for comp in comps]).T
    #pdb.set_trace()
    logpi = np.log(pi)
    logr = logpdfs_c + logpi
    logpdf = logsumexp(logr, axis=1)
    r = np.exp(logr - logpdf[:, None])
    return r, logpdf