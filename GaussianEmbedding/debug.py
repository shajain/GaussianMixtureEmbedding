import matplotlib.pyplot as plt

from misc import sortedplot as sp
from IPython.display import display
import numpy as np
from scipy.stats import norm
import itertools
#from MassSpec.I2GivenI1_NN2.model import I2GivenI1_NN2 as model
import pdb
from  misc.randomSort import randomSort


class Debug:

    def __init__(self):
        self.loss = []
        self.RecLoss = []
        self.GaussianLoss = []
        fig, axs = sp.subplots(3, 3, figsize=(8,8))
        self.fig = fig
        self.axs = axs
        #self.model = model


    def attachData(self, data):
        self.data = data
        #self.x = randomSort(data['x'])[0]
        self.x = data['x']
        #self.dg = dg


    def attachNets(self, autoEncoders):
        self.autoEncoders = autoEncoders


    def lossPlot(self):
        ae = self.autoEncoders[-1]
        loss, recLoss, gaussLoss = ae.loss(self.data)[0:3]
        self.loss = np.append(self.loss, loss)
        #pdb.set_trace()
        sp.sortedplot(self.loss, label='loss', ax=self.axs[0,0])

        self.RecLoss = np.append(self.RecLoss, recLoss)
        sp.sortedplot(self.RecLoss, label='Reconstruction Loss', ax=self.axs[0, 1])

        self.GaussianLoss = np.append(self.GaussianLoss, gaussLoss)
        sp.sortedplot(self.GaussianLoss, label='GaussianLoss', ax=self.axs[0, 2])


    def reconstructionPlot(self):
        xx = self.autoEncoders[-1].reconstruction(self.x)
        intput_dim = self.x.shape[1]
        [sp.sortedplot(self.x[i:i+1], xx[i:i+1], label='current', ax=self.axs[1, 0]) for i in np.arange(intput_dim)]
        # if len(self.autoEncoders) > 1:
        #     xxPrev = self.autoEncoders[-2].reconstruction(self.x)
        #     sp.sortedplot(self.x, xxPrev, label='previous', ax=self.axs[1, 0])

    def histogramPlot(self):
        enc = self.autoEncoders[-1].normalized_encoding(self.data['x'])
        lDims = enc.shape[1]
        ix = np.random.choice(lDims, 2, replace=False)
        [self.axs[1, i+1].hist(enc[:, i], bins=20, density=True, label='normalized encoding') for i in ix]
        xnorm = randomSort(norm.rvs(size=(100, 1)))[0].flatten()
        [self.axs[1,i+1].plot(xnorm, norm.pdf(xnorm)) for i in ix]
        xx, Dens = self.autoEncoders[-1].kdeWrapper(self.data)
        [sp.sortedplot(xx, Dens[i], label='kdefull', ax=self.axs[1, i+1]) for i in np.arange(lDims)]
        batchSize = self.autoEncoders[-1].batchSize
        xx, Dens = self.autoEncoders[-1].kdeWrapper(self.data, batchSize)
        [sp.sortedplot(xx, Dens[i], label='kdeBatch', ax=self.axs[1, i+1]) for i in np.arange(lDims)]
        #self.axs[1, 2].hist(self.data['x'], bins=20, density=True, label='input data')

    def scatterPlot(self):
        enc = self.autoEncoders[-1].normalized_encoding(self.data['x'])
        # Get the number of columns in the matrix
        num_cols = enc.shape[1]
        # Create a dictionary to store the correlation results
        corr_dict = {}
        # Iterate through pairs of columns
        iter  = 0
        for i, j in itertools.combinations(range(num_cols), 2):
            # Compute the correlation between columns i and j
            corr = np.corrcoef(enc[:, i], enc[:, j])[0, 1]
            # Store the correlation result in the dictionary
            corr_dict[(i, j)] = corr
            if iter == 0:
                maxCorKey = (i, j)
                maxCor =  corr
            elif np.abs(corr) > np.abs(maxCor):
                maxCorKey = (i, j)
                maxCor = corr
            iter = iter + 1
        i = maxCorKey[0]
        j = maxCorKey[1]
        self.axs[2, 0].scatter(enc[:, i], enc[:, j])
        self.axs[2, 0].set_title('Dim' + str(i) + ' vs ' + str(j))
        for k in [1, 2]:
            col_indices = np.random.choice(enc.shape[1], size=2, replace=False)
            i = col_indices[0]
            j = col_indices[1]
            self.axs[2, k].scatter(enc[:, i], enc[:, j])
            self.axs[2, k].set_title('Dim' + str(i) + ' vs ' + str(j))





    def afterUpdate(self, loss):
        print('after Update')
        self.lossPlot()
        self.reconstructionPlot()
        self.histogramPlot()
        self.scatterPlot()
        self.displayPlots()
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


