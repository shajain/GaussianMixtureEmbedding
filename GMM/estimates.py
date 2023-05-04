import pdb

import numpy as np


def muEstimate(x,w, w_sum):
    if isinstance(x, list):
        xw_sum = sum([np.sum(xx * ww[:, None], axis=0) for (xx, ww) in zip(x, w)])
    else:
        xw_sum = np.sum(x * w[:,None], axis=0)
    return xw_sum / w_sum


def covEstimate(x, w, w_sum, mu):
    if isinstance(x, list):
        xbar = [xx - mu for xx in x]
        xtx_sum = sum([np.dot(ww * (xx.T), xx) for (xx, ww) in zip(xbar, w)])
    else:
        xbar = x - mu
        xtx_sum = np.dot(w * (xbar.T), xbar)
    return xtx_sum / w_sum


def sphCovEstimate(x, w, w_sum, mu):
    if isinstance(x, list):
        xbarSq = [(xx - mu)**2 for xx in x]
        xwsq_sum = sum([np.sum(xsq * ww[:, None], axis=0) for (xsq, ww) in zip(xbarSq, w)])
    else:
        #pdb.set_trace()
        xwsq_sum  = np.sum(((x-mu)**2)*w[:,None], axis=0)
    return np.diag(xwsq_sum / w_sum)