import tensorflow as tf
import numpy as np
import pdb

def PhiD(d, TF=False):

    def phiD20(s):
        phi = (1+4*s/(2*d-3))**(-0.5)
        return phi

    def phiD2l(s):
        t = s/7.5
        t2 = 3.51562*(t**2)
        t4 = 3.08994*(t**4)
        t6 = 1.20675*(t**6)
        t8 = 0.26597*(t**8)
        t10 = 0.03608*(t**10)
        t12 = 0.00458*(t**12)
        phi = (np.exp(1.0)**(-s/2))*(1 + t2 + t4 + t6 + t8 + t10 + t12)
        #pdb.set_trace()
        return phi

    def phiD2g(s):
        t = 7.5/s
        t0 = 0.398942
        t1 = 0.013286*t
        t2 = 0.002253*(t**2)
        t3 = 0.001576*(t**3)
        t4 = 0.00916*(t**4)
        t5 = 0.020577*(t**5)
        t6 = 0.026355*(t**6)
        t7 = 0.016476*(t**7)
        t8 = 0.003924*(t**8)
        phi = ((2 / s)**0.5) * (t0 + t1 + t2 - t3 + t4 - t5 + t6 - t7 + t8)
        return phi

    # def phiD2TF(s):
    #     #pdb.set_trace()
    #     cond = tf.less(s, 7.5)
    #     def phiD2gWrap():
    #         return phiD2g(s)
    #     def phiD2lWrap():
    #         return phiD2l(s)
    #     phi = tf.cond(cond, phiD2lWrap, phiD2gWrap)
    #     return phi

    def phiD2TF(s):
        #pdb.set_trace()
        safe_s = tf.where(s<=7.5, 7.5, s)
        phi = tf.where(s<=7.5, phiD2l(s), phiD2g(safe_s))
        return phi

    def phiD2(s):
        #pdb.set_trace()
        safe_s = np.where(s <= 7.5, 7.5, s)
        phi = np.where(s <= 7.5, phiD2l(s), phiD2g(safe_s))
        return phi

    if d == 2:
        if TF:
            phiD = phiD2TF
        else:
            #pdb.set_trace()
            phiD = phiD2
    else:
        phiD = phiD20

    return phiD



