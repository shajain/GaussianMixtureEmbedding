import pdb

import numpy as np
from scipy.stats import norm
from scipy.stats import mvn
from sklearn.cluster import KMeans
from GMM.compModel import MVN
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from scipy.special import logsumexp
from data.distributions import mixture
from misc.dictUtils import safeUpdate
from data.distributions import mixture


class GMM:
    def __init__(self, nComps, dim, maxIter=1000, compDist=None, nMix=1, cMemPerSample=None):
        self.nComps = nComps
        self.dim = dim
        self.maxIter = maxIter
        self.iter = 0
        self.nMix = nMix
        #pdb.set_trace()
        if cMemPerSample is None:
            self.cMemPerSample = [np.arange(nComps) for _ in range(nMix)]
        else:
            assert len(cMemPerSample) == nMix
            self.cMemPerSample = cMemPerSample
        self.cMem2SMem(self.cMemPerSample)
        self.mProp = [np.repeat(1/np.size(cMemS), np.size(cMemS)) for cMemS in self.cMemPerSample]
        if compDist is None:
            self.compDist = [MVN(spherical=False) for _ in range(nComps)]
        else:
            self.compDist = compDist
        self.mixture = [mixture([compDist[i] for i in cMemS], mp) for (cMemS, mp)
                        in zip(self.cMemPerSample, self.mProp)]
        self.lls = []
        self.initParRan = False

    def attachDebugger(self, debug):
        self.debug = debug

    def cMem2SMem(self, cMemPerSample):
        self.sMemPerComp = [[] for _ in range(self.nComps)]
        self.cIndPerComp = [[] for _ in range(self.nComps)]
        # Iterate over the membership variable and update the component membership and indexes variables
        for i, mix in enumerate(cMemPerSample):
            for j, component in enumerate(mix):
                self.sMemPerComp[component].append(i)
                self.cIndPerComp[component].append(j)

    def separateData(self, X):
        Fits = [KMeans(n_clusters=len(gci_mix), init='k-means++').fit(x) for (x, gci_mix) in zip(X, self.cMemPerSample)]
        Labels = [fit.labels_ for fit in Fits]
        Centers =[fit.cluster_centers_ for fit in Fits]

        # those component index for which a sample containing it has been processed are stored in comps
        gci_processed = np.array([], dtype='int64')
        centers = np.ones((self.nComps,self.dim))*np.inf
        counts = np.zeros(self.nComps)
        C = []
        for (l, mu, gci_mix) in zip(Labels, Centers, self.cMemPerSample):
            #by the end of the iteration newl will contain updated labels, such that some of the labels are uniquely
            # mapped to the components the sample is supposed to contain and is already present in comps by iteratively
            # finding the closest cluster - component pair. The remaining clusters are given a label corresponding to the
            # component that the sample is suppose to have, but not yet seen in the data processed so far.
            c = np.copy(l)
            # contains global index of the components already processed and also contained in the current sample
            gci_existing = np.intersect1d(gci_mix, gci_processed)
            # contains global index of the unprocessed components contained in the current sample
            gci_new = np.setdiff1d(gci_mix, gci_existing)
            # keeps track of the local component index mapped to already processed components
            lci_assigned = np.array([])
            if gci_existing.shape[0] > 0:
                dist = pairwise_distances(mu, centers)
                #pdb.set_trace()
                dist[:, np.setdiff1d(gci_processed, gci_existing)] = np.inf
                while np.any(np.isfinite(dist)):
                    ix = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
                    lci = ix[0]
                    gci = ix[1]
                    c[l==lci] = gci
                    dist[:, gci] = np.inf
                    dist[lci, :] = np.inf
                    oldCenter = centers[gci,:]
                    oldCount = counts[gci]
                    newCount = oldCount + np.sum(l==lci)
                    newCenter = (oldCenter*oldCount + mu[lci,:] * np.sum(l==lci))/newCount
                    centers[gci,:] = newCenter
                    counts[gci] = newCount
                    lci_assigned = np.hstack((lci_assigned, lci))
            lci_unassigned = np.setdiff1d(np.arange(len(gci_mix)), lci_assigned)
            for (lci, gci) in zip(lci_unassigned, gci_new):
                c[l == lci] = gci
                #pdb.set_trace()
                centers[gci,:] = mu[lci,:]
                counts[gci] = np.sum(l==lci)
                gci_processed = np.hstack((gci_processed, gci))
            C.append(c)
        return X, C

    def initPar(self, X):
        X, C = self.separateData(X)
        #pdb.set_trace()
        [cDist.fit(X, [(c == i).astype('int32') for c in C]) for i, cDist in enumerate(self.compDist)]
        self.mProp = [np.array([np.sum((c == i).astype('int32'))/c.shape[0] for i in cMemS]) for (cMemS, c) in zip(self.cMemPerSample, C)]
        self.initParRan = True
        #pdb.set_trace()

    def refit(self, X, maxIter=None):
        self.iter = 0
        self.fit(X, maxIter)

    def fit(self, X, maxIter=None):
        if not self.initParRan or self.dataOOD(X):
            self.initPar(X)
        if maxIter is None:
            maxIter = self.maxIter
        while self.iter < maxIter:
            self.beforeUpdate()
            self.run(X)
            self.afterUpdate()
            self.iter+=1

    def dataOOD(self, X):
        maxDensPerCmp = np.array([np.max(np.vstack([cmp.pdf(x) for x in  X])) for cmp in self.compDist])
        max = np.max(maxDensPerCmp)
        min = np.min(maxDensPerCmp)
        oodFlag = min/max < 10**-3
        self.oddFlag = oodFlag
        if self.oddFlag:
            print('OOD input to GMM: Reinitializing GMM parameters')
        return oodFlag


    def run(self, X):
        R = self.responsibility(X)
        self.mProp = [np.array([np.sum(r)/x.shape[0] for r in rr.T]) for (rr,x) in zip(R,X)]
        [cDist.fit([X[s] for s in sMem], [R[s][:,c] for (s,c) in zip(sMem, cInd)]) for (cDist, sMem, cInd) in
         zip(self.compDist, self.sMemPerComp, self.cIndPerComp)]

    def responsibility(self, X):
        R = [self.__responsibility__(x, i)[0] for i, x in enumerate(X)]
        return R

    def __responsibility__(self, x, sample_ix):
        logCPdf = np.hstack([self.compDist[j].logpdf(x)[:, None] for j in self.cMemPerSample[sample_ix]])
        logCPdf_w = logCPdf + np.log(self.mProp[sample_ix])
        logMPdf = logsumexp(logCPdf_w, axis=1, keepdims=True)
        R = np.exp(logCPdf_w - logMPdf)
        #R = np.ones((x.shape[0], self.cMemPerSample[sample_ix].size))/self.cMemPerSample[sample_ix].size
        #pdb.set_trace()
        #logMPdf = np.ones((x.shape[0], 1))
        return R, logMPdf



    # def __responsibility__(self, X):
    #     #pdb.set_trace()
    #     logCPdf = [np.hstack([cDist.logpdf(x)[:,None] for cDist in self.compDist]) for x in X]
    #     logCPdf_w = [lPdf[:, cMemS] + np.log(mp) for (lPdf, cMemS, mp) in zip(logCPdf, self.cMemPerSample, self.mProp)]
    #     logMPdf = [logsumexp(lPdf, axis=1, keepdims=True) for lPdf in logCPdf_w]
    #     R = [np.exp(lCPdf - lMPdf) for(lCPdf, lMPdf) in zip(logCPdf_w, logMPdf)]
    #     return R, logMPdf

    def logLikelihood(self, X, equallyWeightedSamples=False):
        logMPdf = [self.__responsibility__(x, i)[1] for i, x in enumerate(X)]
        mixLL = [np.mean(lMPdf) for lMPdf in logMPdf]
        if not equallyWeightedSamples:
            ss = [x.shape[0] for x in X]
            ll = sum([s*mLL for(s, mLL) in zip(ss, mixLL)])/sum([s for s in ss])
        else:
            ll = sum([mLL for(mLL) in zip(mixLL)])
        return ll

    def copy(self):
        compDist = [cDist.copy() for cDist in self.compDist]
        gmm = GMM(self.nComps, self.dim, self.maxIter, compDist, self.nMix, self.cMemPerSample)
        gmm.mProp = np.copy(self.mProp)
        gmm.initParRan = self.initParRan
        return gmm

    def attachDebugger(self, debug):
        self.debug = debug
        self.debug.attachGMM(self)

    def beforeUpdate(self):
        if hasattr(self, 'debug'):
            self.debug.beforeUpdate(self.iter)

    def afterUpdate(self):
        if hasattr(self, 'debug'):
            self.debug.afterUpdate()

