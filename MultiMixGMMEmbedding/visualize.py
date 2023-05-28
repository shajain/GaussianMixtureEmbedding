import matplotlib.pyplot as plt

from misc import sortedplot as sp
from IPython.display import display
import numpy as np
from scipy.stats import norm
from scipy.stats import bernoulli
from GMM.utils import matchComponents
import itertools
from plots.CIEllipse import CIEllipse
#from MassSpec.I2GivenI1_NN2.model import I2GivenI1_NN2 as model
import pdb
from  misc.randomSort import randomSort


class Visualizer:

    class SampleVisualizer:
        def __init__(self, x, sample_ix, vis, dg=None):
            self.x = x
            self.sample_ix = sample_ix
            self.RecLoss = []
            self.vis = vis
            if dg is not None:
                self.dg = dg
                self.RTrue = dg.responsibility(self.x)

        def beforePlotting(self):
            self.ae = self.vis.AEs[-1]
            self.axs = self.vis.axsSmp[self.sample_ix, :]
            self.mae = self.ae.mixtureAEs[self.sample_ix]
            self.comps = self.mae.mixture().comps

        def plot(self):
            self.beforePlotting()
            self.reconstructionLossPlot()
            self.R = self.responsibiltyPlot()
            self.scatterPlot()
            self.gLosses = self.gaussianLoss()
            self.u, self.p = self.kdeMarginals()

        def reconstructionLossPlot(self):
            self.RecLoss.append(self.ae.reconstructionLoss(self.x))
            sp.sortedplot(self.RecLoss, label='Recon Loss', ax=self.axs[0])

        def gaussianLoss(self):
            gLosses = self.mae.GaussianLoss(self.x)[1]
            return gLosses

        # def normEncoding(self, enc):
        #     normEnc = self.mae.normalized_encoding(enc)
        #     return normEnc

        def kdeMarginals(self):
            u, P, _ = self.mae.kdeWrapper(self.x, batchSize=500)
            return u, P

        def responsibiltyPlot(self):
            R = self.mae.responsibility(self.x)
            #pdb.set_trace()
            permTrue, perm = matchComponents(self.x, self.RTrue, R)
            [self.axs[1].scatter(self.RTrue[:, pT], R[:, p], s=5, alpha=0.1, label='resp') for (pT, p) in zip(permTrue, perm)]
            return R

        def scatterPlot(self):
            R = self.mae.responsibility(self.x)
            self.__scatter__(self.x, self.RTrue, 2)
            self.axs[2].set_title("true")
            self.__scatter__(self.x, R, 3)
            self.axs[3].set_title("predicted")

        def __scatter__(self, x, R, axs_ix):
            enc = self.ae.encoding(x)
            Enc = [enc[bernoulli.rvs(r).astype('bool'), :] for r in R.T]
            [self.axs[axs_ix].scatter(enc[:,self.vis.xcol], enc[:,self.vis.ycol], s=5, alpha=0.1, label='encoding') for enc in Enc]
            [CIEllipse(cmp.mu, cmp.cov, ax=self.axs[axs_ix], edgecolor='k') for cmp in self.comps]
            self.axs[axs_ix].set_xlabel('dim: ' + str(self.vis.xcol))
            self.axs[axs_ix].set_ylabel('dim: ' + str(self.vis.ycol))


    class ComponentVisualizer:
        def __init__(self, comp_ix, vis):
            self.comp_ix = comp_ix
            self.GLosses_c = []
            self.vis = vis

        def plot(self):
            self.beforePlotting()
            self.gaussianLossPlot()
            self.scatterPlot()
            self.kdePlot()

        def beforePlotting(self):
            ae = self.vis.AEs[-1]
            R = self.vis.R
            RTrue = self.vis.RTrue
            self.axs = self.vis.axsComp[self.comp_ix, :]
            sMemC = ae.GMM.sMemPerComp[self.comp_ix]
            cIndC = ae.GMM.cIndPerComp[self.comp_ix]
            R_c = np.hstack([R[s][:,c:(c+1)] for (s,c) in zip(sMemC, cIndC)])
            RTrue_c = np.hstack([RTrue[s][:, c:(c + 1)] for (s, c) in zip(sMemC, cIndC)])
            Enc = [self.vis.Enc[s] for s in sMemC]
            self.Enc_c = [enc[bernoulli.rvs(r).astype('bool'), :] for (enc, r) in zip(Enc, R_c.T)]
            self.EncTrue_c = [enc[bernoulli.rvs(r).astype('bool'), :] for (enc, r) in zip(Enc, RTrue_c.T)]
            self.U_c = [self.vis.U[s] for s in sMemC]
            self.P_c = [self.vis.P[s][c] for (s, c) in zip(sMemC, cIndC)]
            if len(self.GLosses_c) == 0:
                self.GLosses_c = [[self.vis.GLosses[s][c]] for (s, c) in zip(sMemC, cIndC)]
            else:
                [lst.append(self.vis.GLosses[s][c]) for (lst, s, c) in zip(self.GLosses_c, sMemC, cIndC)]
            if np.size(sMemC) > 1:
                self.rSample_ix = np.random.choice(sMemC, size=2, replace=False)
            else:
                self.rSample_ix = sMemC
            self.comp = ae.GMM.compDist[self.comp_ix]



        def gaussianLossPlot(self):
            [sp.sortedplot(losses, label='Gaussian Loss', ax=self.axs[0]) for losses in self.GLosses_c]

        def scatterPlot(self):
            [self.__scatterPlot__(i+1, enc=self.EncTrue_c[s_ix], label = 'True('+str(s_ix)+')')
             for i, s_ix in enumerate(self.rSample_ix)]
            [self.__scatterPlot__(i+1, enc=self.Enc_c[s_ix], label='Predicted(' + str(s_ix) + ')')
             for i, s_ix in enumerate(self.rSample_ix)]
            [CIEllipse(self.comp.mu, self.comp.cov, ax=self.axs[i+1], edgecolor='k') for i in range(self.rSample_ix.size)]

        def __scatterPlot__(self, axs_ix, enc, label):
            self.axs[axs_ix].scatter(enc[:, self.vis.xcol], enc[:, self.vis.ycol], s=5, alpha=0.1, label=label)
            self.axs[axs_ix].set_xlabel('dim ' + str(self.vis.xcol))
            self.axs[axs_ix].set_ylabel('dim ' + str(self.vis.ycol))

        def kdePlot(self):
            self.__kdePlot__(3, self.vis.xcol)
            self.__kdePlot__(4, self.vis.ycol)

        def __kdePlot__(self, axs_ix, col):
            [sp.sortedplot(u, P[col],  ax=self.axs[axs_ix]) for (u,P) in zip(self.U_c, self.P_c)]
            [sp.sortedplot(u, norm.pdf(u), ax=self.axs[axs_ix]) for u in self.U_c]
            self.axs[axs_ix].set_xlabel('dim ' + str(col))



    def __init__(self, X, nComps, DG=None):
        self.X = X
        self.nComps = nComps
        self.nMix = len(X)
        if DG is not None:
            self.DG = DG
        else:
            self.DG = [None for _ in range(self.nMix)]
        self.sVis = [Visualizer.SampleVisualizer(x, s, self, dg) for s, (x, dg) in enumerate(zip(X, DG))]
        self.cVis = [Visualizer.ComponentVisualizer(c, self) for c in range(nComps)]
        # self.figSmp, self.axsSmp = sp.subplots(self.nMix, 4, figsize=(10, 12))
        # self.figComp, self.axsComp = sp.subplots(self.nComps, 5, figsize=(10, 12))
        #self.model = model

    def attachNets(self, nets):
        self.AEs = nets

    def plot(self):
        ae = self.AEs[-1]
        self.Enc = [ae.encoding(x) for x in self.X]
        lDim = ae.latent_dim
        if lDim > 1:
            rDim = np.random.choice(lDim, size=2, replace=False)
            self.xcol = np.min(rDim)
            self.ycol = np.max(rDim)
        else:
            self.xcol = 0
            self.ycol = 0

        self.figSmp, self.axsSmp = sp.subplots(self.nMix, 4, figsize=(10, 5), squeeze=False)
        self.figComp, self.axsComp = sp.subplots(self.nComps, 5, figsize=(10, 5), squeeze=False)
        [sVis.plot() for sVis in self.sVis]
        self.R = [sVis.R for sVis in self.sVis]
        self.RTrue = [sVis.RTrue for sVis in self.sVis]
        self.GLosses = [sVis.gLosses for sVis in self.sVis]
        self.U = []
        self.P = []
        for sVis in self.sVis:
            u, P = sVis.kdeMarginals()
            self.U.append(u)
            self.P.append(P)

        [cVis.plot() for cVis in self.cVis]

        for axs in np.hstack([self.axsSmp.reshape(-1), self.axsComp.reshape(-1)]):
            axs.legend( )
        display(self.figSmp)
        display(self.figComp)
        for axs in np.hstack([self.axsSmp.reshape(-1), self.axsComp.reshape(-1)]):
            axs.clear( )


    def afterUpdate(self):
        print('after Update')
        self.plot()
        # sp.show()

    def beforeUpdate(self, iter):
        print('iteration ' + str(iter))
        return

    def beforeTraining(self):
        return