from misc import sortedplot as sp
from IPython.display import display
import numpy as np
from scipy.stats import norm
from scipy.stats import uniform as unif
#from MassSpec.I2GivenI1_NN2.model import I2GivenI1_NN2 as model
from matplotlib.patches import Ellipse
from KDE.kde import kde
from KDE.kde import bandwidth
import pdb
from  misc.randomSort import randomSort


class Debug:

    def __init__(self):
        self.loss = []
        self.recLoss_c = None
        self.gaussianLoss_c = None
        self.corrLoss_c = None
        fig, axs = sp.subplots(6, 4, figsize=(8,8))
        self.fig = fig
        self.axs = axs
        #self.model = model


    def attachData(self, data):
        self.data = data
        self.x = data['x']

    def attachModel(self, trueModel):
        self.trueModel = trueModel



    def attachNets(self, autoEncoders):
        self.autoEncoders = autoEncoders


    def lossPlot(self):
        loss, recLoss_c, gaussianLoss_c, corrLoss_c = self.autoEncoders[-1].loss(self.data)
        self.loss.append(loss)
        if self.recLoss_c is None:
            self.recLoss_c = [[recLoss] for recLoss in recLoss_c ]
            self.gaussianLoss_c = [[gLoss] for gLoss in gaussianLoss_c]
            self.corrLoss_c = [[cLoss] for cLoss in corrLoss_c]
        else:
            [self.recLoss_c[i].append(recLoss) for (i, recLoss) in enumerate(recLoss_c)]
            [self.gaussianLoss_c[i].append(gLoss) for (i, gLoss) in enumerate(gaussianLoss_c)]
            [self.corrLoss_c[i].append(cLoss) for (i, cLoss) in enumerate(corrLoss_c)]
        #pdb.set_trace()
        sp.sortedplot(self.loss, label='loss', ax=self.axs[0, 0])
        [sp.sortedplot(self.recLoss_c[i], label=str(i), ax=self.axs[0, 1]) for i in np.arange(len(self.recLoss_c))]
        self.axs[0,1].set_title('Reconstruction Loss')
        [sp.sortedplot(self.gaussianLoss_c[i], label=str(i), ax=self.axs[0, 2]) for i in np.arange(len(self.gaussianLoss_c))]
        self.axs[0, 2].set_title('Gaussian Loss')
        [sp.sortedplot(self.corrLoss_c[i], label=str(i), ax=self.axs[0, 3]) for i in np.arange(len(self.corrLoss_c))]
        self.axs[0, 3].set_title('Correlation Loss')

    def reconstructionPlot(self):
        fit = self.autoEncoders[-1]
        xx = fit.reconstruction(self.x)
        nDims = self.x.shape[1]
        if nDims > 2:
            self.rDims = np.random.choice(nDims, 2, replace=False)
        else:
            self.rDims = np.arange(nDims)
        [self.axs[1, i].scatter(self.x[:, dim:dim+1], xx[:, dim:dim+1], label='D' + str(dim)) for (i, dim) in enumerate(self.rDims)]
        [self.axs[1, i].set_title('D' + str(d)) for (i, d) in enumerate(self.rDims)]
        cSamples = fit.componentSamples(self.x, fit.responsibility_x(self.x), 500)
        nCmp = len(cSamples)
        if nCmp > 2:
            rCmp = np.random.choice(nCmp, 2, replace=False)
        else:
            rCmp = np.arange(nCmp)
        [[self.axs[1, i+2].scatter(cSamples[c][:, dim:dim + 1], fit.reconstruction(cSamples[c])[:, dim:dim + 1], label='D' + str(dim)) for dim in
         np.arange(nDims)] for (i, c) in enumerate(rCmp)]
        [self.axs[1, i+2].set_title('C' + str(c)) for (i, c) in enumerate(rCmp)]
        # if len(self.autoEncoders) > 1:
        #     xxPrev = self.autoEncoders[-2].reconstruction(self.x)
        #     sp.sortedplot(self.x, xxPrev, label='previous', ax=self.axs[1, 0])

    def histogramPlot(self, row=2):
        fit = self.autoEncoders[-1]
        #pdb.set_trace()
        enc = fit.encoding(self.x)
        u_c, probs_c_dim, probs_c, u_m, probs_m_dim, probs_m = fit.kdeWrapper(self.data, 500)
        # nDims = enc.shape[1]
        # rDims = np.random.choice(nDims,2, replace=False)
        means = fit.means
        stds = fit.stds

        #pdb.set_trace()
        [self.axs[row, i*2].hist(enc[:, d], bins=50, density=True, label='enc') for (i, d) in enumerate(self.rDims)]
        #pdb.set_trace()
        [sp.sortedplot(u_m[:, d:d+1], probs_m_dim[:, d:d+1], ax=self.axs[row, i*2]) for (i, d) in enumerate(self.rDims)]
        u = fit.randomUniform()
        [[sp.sortedplot(u*std[d] + mean[d], (1/std[d])*norm.pdf(u), ax=self.axs[row, (2*i+1)]) for (i, d) in enumerate(self.rDims)]
            for (mean, std) in zip(means, stds)]
        [[sp.sortedplot(u[:, d:d+1], p[:, d:d+1], ax=self.axs[row, (2*i+1)]) for (i, d) in enumerate(self.rDims)]
            for (u, p) in zip(u_c, probs_c_dim)]
        [self.axs[row, (2*i+1)].set_title('D' + str(d)) for (i, d) in enumerate(self.rDims)]
        #self.axs[2, 2].hist(self.data['x'], bins=20, density=True, label='input data')

    def histogramPlot2(self, row=5):
        fit = self.autoEncoders[-1]
        orths = fit.orths
        #pdb.set_trace()
        size =500
        R = fit.responsibility_x(self.x)
        cSamples = fit.componentSamples(self.x, R, size)
        nCmps = len(cSamples)
        bws = [max(bandwidth(size), bandwidth(sum(r))) for r in R.T]
        enc_ns = fit.normalized_encodings(cSamples)
        encT_ns = [np.matmul(enc_n - np.mean(enc_n, axis=0), orth) for (enc_n, orth) in zip(enc_ns, orths)]
        nDims = enc_ns[0].shape[1]
        [self.axs[row, 2*i].scatter(enc_n[:, 0:1], enc_n[:, 1:2], label='C'+str(i)) for (i, enc_n) in enumerate(encT_ns)]
        [[self.axs[row, 2*i].plot([0, o[0]], [0, o[1]], color='k') for o in orth.T] for (i, orth) in enumerate(orths)]
        #encT_ns = [np.matmul(enc_n, orth) for (enc_n, orth) in zip(enc_ns, orths)]
        kdeFncs = [[kde(encT_n[:, d:d + 1], bw)[0] for d in np.arange(fit.latent_dim)] for (encT_n, bw) in
                   zip(encT_ns, bws)]
        u = fit.randomUniform()
        [sp.sortedplot(u, norm.pdf(u), ax=self.axs[row, (2 * i + 1)]) for (i, c) in enumerate(np.arange(nCmps))]
        [[sp.sortedplot(u, kdeFnc[d](u), label= 'D'+ str(d), ax=self.axs[row, (2 * i + 1)]) for d in
          np.arange(nDims)] for (i, kdeFnc) in enumerate(kdeFncs)]
        [self.axs[row, (2*i+1)].set_title('C' + str(c)) for (i, c) in enumerate(np.arange(2))]


    def posteriorPlot(self):
        fit = self.autoEncoders[-1]
        p = fit.posterior(self.x)
        p_true = self.trueModel.posterior(self.x)
        #pdb.set_trace()
        # ix = np.argsort(p_true)
        # p_true = p_true[ix]
        # p = p[ix]
        # sp.sortedplot(p_true, p, ax=self.axs[3, 0])
        # sp.sortedplot(p_true, 1-p, ax=self.axs[3, 0])
        self.axs[3, 0].scatter(p_true, p)
        #self.axs[3, 0].scatter(p_true, 1-p)
        self.axs[3, 0].set_title('unlabeled')
        cSamples = fit.componentSamples(self.x, fit.responsibility_x(self.x), 500)
        p_c = [fit.posterior(x) for x in cSamples]
        p_true_c = [self.trueModel.posterior(x) for x in cSamples]
        [self.axs[3, 1].scatter(p_true, p, label='C' + str(c)) for (c, (p_true, p)) in enumerate(zip(p_true_c, p_c))]
        [self.axs[3, 2].scatter(p_true, 1-p, label='C' + str(c)) for (c, (p_true, p)) in enumerate(zip(p_true_c, p_c))]

    def GMMPlot(self):
        fit = self.autoEncoders[-1]
        means = fit.means
        stds = fit.stds
        #pi = fit.pi
        ix = np.random.choice(self.x.shape[0], 100)
        enc = fit.encoding(self.x[ix,:])
        self.axs[4, 0].scatter(enc[:, 0:1], enc[:, 1:2])
        [self.axs[4, 0].scatter(mean[0], mean[1], s=40) for mean in means]
        ellipse = []
        for (mean, std) in zip(means, stds):
            ellipse.append(Ellipse(mean[0:2], 4*std[0], 4*std[1], fill=False))
        [self.axs[4, 0].add_patch(e) for e in ellipse]

        cSamples = fit.componentSamples(self.x, fit.responsibility_x(self.x), 100)
        encs = [fit.encoding(csmp) for csmp in cSamples]
        [self.axs[4, 1].scatter(enc[:, 0:1], enc[:, 1:2]) for (i,enc) in enumerate(encs)]
        [self.axs[4, 1].scatter(mean[0], mean[1], s=40) for mean in means]
        ellipse = []
        for (mean, std) in zip(means, stds):
            ellipse.append(Ellipse(mean[0:2], 4 * std[0], 4 * std[1], fill=False))
        [self.axs[4, 1].add_patch(e) for e in ellipse]


    def afterUpdate(self, loss):
        print('after Update')
        self.lossPlot()
        self.reconstructionPlot()
        self.histogramPlot()
        self.posteriorPlot()
        self.GMMPlot()
        self.histogramPlot2(5)
        self.displayPlots()

        sp.subplots_adjust(hspace=0.4)
        # sp.show()

    def beforeUpdate(self, iter):
        if np.remainder(iter, 10) == 0:
            print('Iteration' + str(iter))
        return

    def beforeTraining(self, par):
        print('before Training')
        #self.plotllHistory(par)
        return


    def displayPlots(self):
        for axs in self.axs.reshape(-1):
            #pdb.set_trace()
            axs.legend( )
        display(self.fig)
        sp.close( )
        for axs in self.axs.reshape(-1):
            axs.clear( )
        fig, axs = sp.subplots(6, 4, figsize=(8, 8))
        self.fig = fig
        self.axs = axs


