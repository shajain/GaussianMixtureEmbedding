from misc import sortedplot as sp
from IPython.display import display
import numpy as np
from scipy.stats import norm
from scipy.stats import uniform as unif
#from MassSpec.I2GivenI1_NN2.model import I2GivenI1_NN2 as model
import pdb
from  misc.randomSort import randomSort


class Debug:

    def __init__(self):
        self.loss = []
        self.recLoss_c = None
        self.gaussianLoss_c = None
        fig, axs = sp.subplots(3, 3, figsize=(8,8))
        self.fig = fig
        self.axs = axs
        #self.model = model


    def attachData(self, data):
        self.data = data
        self.x = randomSort(data['x'])[0]

    def attachModel(self, trueModel):
        self.trueModel = trueModel



    def attachNets(self, autoEncoders):
        self.autoEncoders = autoEncoders


    def lossPlot(self):
        loss, recLoss_c, gaussianLoss_c = self.autoEncoders[-1].loss(self.data)
        self.loss.append(loss)
        if self.recLoss_c is None:
            self.recLoss_c = [[recLoss] for recLoss in recLoss_c ]
            self.gaussianLoss_c = [[gLoss] for gLoss in gaussianLoss_c]
        else:
            [self.recLoss_c[i].append(recLoss) for (i, recLoss) in enumerate(recLoss_c)]
            [self.gaussianLoss_c[i].append(gLoss) for (i, gLoss) in enumerate(gaussianLoss_c)]
        #pdb.set_trace()
        sp.sortedplot(self.loss, label='loss', ax=self.axs[0, 0])
        [sp.sortedplot(self.recLoss_c[i], label='Reconstruction Loss', ax=self.axs[0, 1]) for i in np.arange(len(self.recLoss_c))]
        [sp.sortedplot(self.gaussianLoss_c[i], label='Gaussian Loss', ax=self.axs[0, 2]) for i in np.arange(len(self.gaussianLoss_c))]


    def reconstructionPlot(self):
        xx = self.autoEncoders[-1].reconstruction(self.x)
        sp.sortedplot(self.x, xx, label='current', ax=self.axs[2, 0])
        if len(self.autoEncoders) > 1:
            xxPrev = self.autoEncoders[-2].reconstruction(self.x)
            sp.sortedplot(self.x, xxPrev, label='previous', ax=self.axs[2, 0])

    def histogramPlot(self):
        fit = self.autoEncoders[-1]
        #pdb.set_trace()
        enc = fit.encoding(self.x)
        u_c, probs_c, u_m, probs_m = fit.kdeWrapper(self.data, 500)
        means = fit.means
        stds = fit.stds
        #pdb.set_trace()
        self.axs[1,0].hist(enc, bins=20, density=True, label='encoding')
        sp.sortedplot(u_m, probs_m, ax=self.axs[1, 0])
        u = fit.randomUniform()
        [sp.sortedplot(u*std + mean, (1/std)*norm.pdf(u), label='Normal', ax=self.axs[1, 1]) for (mean, std) in zip(means, stds)]
        [sp.sortedplot(u, p, label='kde', ax=self.axs[1, 1]) for (u, p) in zip(u_c, probs_c)]
        self.axs[1, 2].hist(self.data['x'], bins=20, density=True, label='input data')

    def posteriorPlot(self):
        fit = self.autoEncoders[-1]
        p = fit.posterior(self.x)
        #pdb.set_trace()
        self.axs[2, 1].plot(self.x.flatten(), p, label= 'Est. posterior')
        self.axs[2, 1].plot(self.x.flatten(), 1-p, label='Est. 1-posterior')
        self.axs[2, 1].plot(self.x.flatten(), self.trueModel.posterior(self.x), label= 'True posterior')



    def afterUpdate(self, loss):
        print('after Update')
        self.lossPlot()
        self.reconstructionPlot()
        self.histogramPlot()
        self.posteriorPlot()
        self.displayPlots()
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
        fig, axs = sp.subplots(3, 3, figsize=(8, 8))
        self.fig = fig
        self.axs = axs


