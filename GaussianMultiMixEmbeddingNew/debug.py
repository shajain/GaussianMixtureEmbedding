import matplotlib.pyplot as plt

from misc import sortedplot as sp
from IPython.display import display
import numpy as np
from scipy.stats import norm
from scipy.stats import bernoulli
from GaussianMixEmbedding.debug import Debug as DBBase

import itertools
#from MassSpec.I2GivenI1_NN2.model import I2GivenI1_NN2 as model
import pdb
from  misc.randomSort import randomSort


# class Debug:
#
#     def __init__(self, nMix=2):
#         self.DB = [DBBase() for _ in range(nMix)]
#
#         #self.model = model
#
#
#     def attachData(self, data, DG):
#         self.data = data
#         self.X = data['X']
#         self.DG = DG
#         self.postTrue = [dg.pn_posterior(self.x) for (x, dg) in zip(self.X, DG)]
#         self.RTrue = [dg.responsibility(self.x) for (x, dg) in zip(self.X, DG)]
#
#
#     def attachNets(self, autoEncoders):
#         self.autoEncoders = autoEncoders
#
#
#
#     def lossPlot(self):
#         ae = self.autoEncoders[-1]
#         loss, recLoss, gaussLoss = ae.loss(self.data)[0:3]
#         self.loss = np.append(self.loss, loss)
#         #pdb.set_trace()
#         sp.sortedplot(self.loss, label='loss', ax=self.axs[0,0])
#
#         self.RecLoss = np.append(self.RecLoss, recLoss)
#         sp.sortedplot(self.RecLoss, label='Reconstruction Loss', ax=self.axs[0, 1])
#
#         self.GaussianLoss = np.append(self.GaussianLoss, gaussLoss)
#         sp.sortedplot(self.GaussianLoss, label='GaussianLoss', ax=self.axs[0, 2])
#         #pdb.set_trace()
#         R = np.hstack([r[:, np.newaxis] for r in  self.RTrue[0] + self.RTrue[1]])
#         self.GaussianLossTrue = np.append(self.GaussianLossTrue, ae.loss(self.data, R=R)[2])
#         sp.sortedplot(self.GaussianLossTrue, label='GaussianLossTrue', ax=self.axs[0, 2])
#
#
#     def reconstructionPlot(self):
#         xx = self.autoEncoders[-1].reconstruction(self.x)
#         input_dim = xx.shape[1]
#         [sp.sortedplot(self.x[i:i+1], xx[i:i+1], label='current', ax=self.axs[0, 3]) for i in np.arange(input_dim)]
#         # if len(self.autoEncoders) > 1:
#         #     xxPrev = self.autoEncoders[-2].reconstruction(self.x)
#         #     sp.sortedplot(self.x, xxPrev, label='previous', ax=self.axs[1, 0])
#
#     def gaussianPlot(self):
#         ae = self.autoEncoders[-1]
#         size = ae.batchSize
#         # x = self.data['x']
#         # R = ae.responsibility(x)
#         # Enc1 = [ae.normalized_encoding(x, r) for r in R.T]
#         u, probs, Enc = self.autoEncoders[-1].kdeWrapper(self.data, size=size)
#         lDims = self.autoEncoders[-1].latent_dim
#         for (i,(enc, p))in enumerate(zip(Enc, probs)):
#             dims = np.random.choice(lDims, 2, replace=False)
#             ix = dims[0]
#             iy = dims[1]
#             xx = enc[:, ix]
#             yy = enc[:, iy]
#             self.axs[1+i, 0].scatter(xx, yy)
#             self.axs[1+i, 0].set_xlabel('dim: ' + str(ix))
#             self.axs[1+i, 0].set_ylabel('dim: ' + str(iy))
#             sp.sortedplot(u, p[ix], ax=self.axs[1+i, 1])
#             sp.sortedplot(u, norm.pdf(u), ax=self.axs[1+i, 1])
#             self.axs[1+i, 1].set_xlabel('dim: ' + str(ix))
#             sp.sortedplot(u, p[iy], ax=self.axs[1+i, 2])
#             sp.sortedplot(u, norm.pdf(u), ax=self.axs[1+i, 2])
#             self.axs[1+i, 2].set_xlabel('dim: ' + str(iy))
#
#     # def scatterPlot(self):
#     #     enc = self.autoEncoders[-1].normalized_encoding(self.data['x'])
#     #     # Get the number of columns in the matrix
#     #     num_cols = enc.shape[1]
#     #     # Create a dictionary to store the correlation results
#     #     corr_dict = {}
#     #     # Iterate through pairs of columns
#     #     iter  = 0
#     #     for i, j in itertools.combinations(range(num_cols), 2):
#     #         # Compute the correlation between columns i and j
#     #         corr = np.corrcoef(enc[:, i], enc[:, j])[0, 1]
#     #         # Store the correlation result in the dictionary
#     #         corr_dict[(i, j)] = corr
#     #         if iter == 0:
#     #             maxCorKey = (i, j)
#     #             maxCor =  corr
#     #         elif np.abs(corr) > np.abs(maxCor):
#     #             maxCorKey = (i, j)
#     #             maxCor = corr
#     #         iter = iter + 1
#     #     i = maxCorKey[0]
#     #     j = maxCorKey[1]
#     #     self.axs[2, 0].scatter(enc[:, i], enc[:, j])
#     #     self.axs[2, 0].set_title('Dim' + str(i) + ' vs ' + str(j))
#     #     for k in [1, 2]:
#     #         col_indices = np.random.choice(enc.shape[1], size=2, replace=False)
#     #         i = col_indices[0]
#     #         j = col_indices[1]
#     #         self.axs[2, k].scatter(enc[:, i], enc[:, j])
#     #         self.axs[2, k].set_title('Dim' + str(i) + ' vs ' + str(j))
#
#     def scatterPlot(self):
#         lDims = self.autoEncoders[-1].latent_dim
#         dims = np.random.choice(lDims, (2,2), replace=True)
#         x = self.data['x']
#         n = x.shape[0]
#         ae = self.autoEncoders[-1]
#         size = ae.batchSize
#         ix = np.random.choice(n, size, replace=True)
#         x = x[ix, :]
#         enc = ae.encoding(x)
#
#         RTrue = self.dg.responsibility(x)
#         IXTrue = [bernoulli.rvs(r).astype('bool').flatten() for r in RTrue[0] + RTrue[1]]
#
#         R = ae.responsibility(x)
#         IX = [bernoulli.rvs(r).astype('bool').flatten() for r in R.T]
#         #pdb.set_trace()
#
#         for (i, d) in enumerate(dims):
#             d0 = d[0]
#             d1 = d[1]
#             #pdb.set_trace()
#             [self.axs[self.nComps+1, 2*i].scatter(enc[ix, d0], enc[ix, d1]) for ix in IXTrue]
#             self.axs[self.nComps+1, 2 * i].set_title("true")
#             [self.axs[self.nComps+1, 2*i+1].scatter(enc[ix, d0], enc[ix, d1]) for ix in IX]
#             self.axs[self.nComps+1, 2*i+1].set_title("predicted")
#             [self.axs[self.nComps+1, 2*i + k].set_xlabel('dim: ' + str(d0)) for k in [0,1]]
#             [self.axs[self.nComps+1, 2*i+k].set_ylabel('dim: ' + str(d1)) for k in [0,1]]
#
#
#
#
#     def afterUpdate(self, loss):
#         print('after Update')
#         self.nComps = self.autoEncoders[-1].nComps
#         fig, axs = sp.subplots(self.nComps + 2, 4, figsize=(10, 12))
#         self.fig = fig
#         self.axs = axs
#
#         self.lossPlot()
#         self.reconstructionPlot()
#         self.gaussianPlot()
#         self.scatterPlot()
#         self.displayPlots()
#         # sp.show()
#
#     def beforeUpdate(self, iter):
#         if np.remainder(iter, 10) == 0:
#             print('Iteration' + str(iter))
#         return
#
#     def beforeTraining(self, par):
#         print('before Training')
#         self.plotllHistory(par)
#         return
#
#
#     def displayPlots(self):
#         for axs in self.axs.reshape(-1):
#             #pdb.set_trace()
#             axs.legend( )
#         display(self.fig)
#         sp.close( )
#         for axs in self.axs.reshape(-1):
#             axs.clear( )
#         # fig, axs = sp.subplots(self.nComps + 2, 4, figsize=(10, 12))
#         # self.fig = fig
#         # self.axs = axs
#



class Debug:

    def __init__(self, nMix=2):
        self.DB = [DBBase() for _ in range(nMix)]

        #self.model = model


    def attachData(self, data, DG):
        self.X = data['X']
        [db.attachData({'x':x}, dg) for (db, x, dg) in zip(self.DB, self.X, DG)]


    def attachNets(self, autoEncoders):
        self.autoEncoders = autoEncoders



    def lossPlot(self, db, ae):
        db.lossPlot(ae)


    def reconstructionPlot(self, db, ae):
        db.reconstructionPlot(ae)

    def gaussianPlot(self, db, ae):
        db.gaussianPlot(ae)


    def scatterPlot(self, db, ae):
        db.scatterPlot(ae)




    def afterUpdate(self):
        print('after Update')
        #pdb.set_trace()
        for i, (db, ae) in enumerate(zip(self.DB, self.autoEncoders[-1].gMixAE)):
            print('Mixture:' + str(i))
            db.afterUpdate(ae)
        # AE = self.autoEncoders[-1]
        # self.FIG = []
        # self.AXS = []
        # for (db, ae) in zip(self.DB, AE.gMixAE):
        #     fig, axs = sp.subplots(self.nComps + 2, 4, figsize=(10, 12))
        #     self.FIG.append(fig)
        #     self.AXS.append(axs)
        #     self.nComps = ae.nComps
        #     self.lossPlot()
        #     self.reconstructionPlot()
        #     self.gaussianPlot()
        #     self.scatterPlot()
        #     self.displayPlots()
        # sp.show()

    def beforeUpdate(self, iter):
        if np.remainder(iter, 10) == 0:
            print('Iteration' + str(iter))
        return

    def beforeTraining(self, par):
        print('before Training')
        self.plotllHistory(par)
        return


    def displayPlots(self):
        [db.displayPlots() for db in self.DB]


        # for (Fig, Axs) in zip(self.FIG, self.AXS):
        #     for axs in Axs.reshape(-1):
        #         #pdb.set_trace()
        #         axs.legend( )
        #     display(Fig)
        #     sp.close( )
        # for Axs in self.AXS:
        #     for axs in Axs.reshape(-1):
        #         #pdb.set_trace()
        #         axs.clear( )

