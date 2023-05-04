from scipy.stats import norm
from scipy.stats import uniform
from random import seed
from random import randint
from random import random
import numpy as np
from data.distributions import mixture
from sklearn.datasets import make_spd_matrix as spd
from scipy.stats import dirichlet
from sklearn import metrics
from scipy.stats import multivariate_normal as mvn
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import pdb as pdb
from sklearn.utils import shuffle
from types import SimpleNamespace as SN
from data.datagen import MVNormalMixDG
from data.datagen import DataGenerator
from itertools import combinations


class NormalMixRandomParameters:

    def __init__(self, dim, max_comps):
        self.dim = dim
        self.max_comps = max_comps
        self.n_comps_pos = randint(1, max_comps)
        self.n_comps_neg = randint(1, max_comps)
        self.mu_pos = [np.array([2 * random() - 1 for i in np.arange(dim)]) for j in np.arange(self.n_comps_pos)]
        self.mu_neg = [np.array([2 * random() - 1 for i in np.arange(dim)]) for j in np.arange(self.n_comps_neg)]
        self.sig_pos = [spd(dim) for j in np.arange(self.n_comps_pos)]
        self.sig_neg = [spd(dim) for j in np.arange(self.n_comps_neg)]
        self.p_pos = dirichlet(np.ones(self.n_comps_pos)).rvs([])
        self.p_neg = dirichlet(np.ones(self.n_comps_neg)).rvs([])
        self.alpha = random()

    def computePNDataMetrics(self):
        epsilon = 10 ** -7
        n = 10000
        _, x, y, pos, neg, dg = self.generatePNData(n, n)
        posterior_pos = dg.pn_posterior_balanced(pos)
        irreducibility = np.mean(np.cast['int32'](posterior_pos > 1-epsilon).flatten())
        posterior_x = dg.pn_posterior_balanced(x)
        fpr, tpr, thresholds = metrics.roc_curve(y, posterior_x)
        aucpn = metrics.auc(fpr, tpr)
        #pdb.set_trace()
        return {'aucpn': aucpn, 'irreducibility': irreducibility}

    def createDataGenerator(self, n_pos, n_ul):
        dg = MVNormalMixDG(self.mu_pos, self.sig_pos, self.p_pos, self.mu_neg, self.sig_neg, self.p_neg, self.alpha,
                         n_pos, n_ul)
        return dg

    def generatePNData(self, n_pos, n_neg):
        dg = self.createDataGenerator(500, 2000)
        pos = dg.data_pos(n_pos)
        neg = dg.data_neg(n_neg)
        y = np.concatenate((np.ones([n_pos, 1]), np.zeros([n_neg, 1])), axis=0)
        x = np.concatenate((pos, neg), axis=0)
        xy = np.concatenate((x, y), axis=1)
        return xy, x, y, pos, neg, dg

    def perturb2Irreducibility(self, irr_range):
        metricsPN = self.computePNDataMetrics()
        if irr_range[0] <= metricsPN['irreducibility'] <= irr_range[1]:
            return
        sigma_flag = random( ) > 0.5
        if metricsPN['irreducibility'] < irr_range[0]:
            if sigma_flag:
                self.decrease_covar(irr_range)
            else:
                self.move_mean_out(irr_range)
        else:
            if sigma_flag:
                self.increase_covar(irr_range)
            else:
                if self.equalMeans():
                    self.alignCovar(irr_range)
                else:
                    self.move_mean_in(irr_range)
        metricsPN = self.computePNDataMetrics()
        if irr_range[0] <= metricsPN['irreducibility'] <= irr_range[1]:
            return
        else:
            self.perturb2Irreducibility(irr_range)

    def increase_covar(self, irr_range):
        i_pos = randint(0, self.n_comps_pos - 1)
        sig = self.sig_pos[i_pos]
        sig_ratio = self.sigmaRatio(sig)
        dim_float = np.cast['float32'](self.dim)
        if all(sig_ratio <= 2):
            up = 1.1
            while up > 1.001:
                self.sig_pos[i_pos] = up * sig
                metricsPN = self.computePNDataMetrics()
                if metricsPN['irreducibility'] < irr_range[0]:
                    up = 1 + (up-1)/2
                else:
                    break

    def decrease_covar(self, irr_range):
        i_pos = randint(0, self.n_comps_pos - 1)
        sig = self.sig_pos[i_pos]
        sig_ratio = self.sigmaRatio(sig)
        dim_float = np.cast['float32'](self.dim)
        if all(sig_ratio > 0.5):
            up = 0.5
            while up < .99:
                self.sig_pos[i_pos] = up * sig
                metricsPN = self.computePNDataMetrics( )
                if metricsPN['irreducibility'] > irr_range[1]:
                    up = 1 - (1 - up) / 2
                else:
                    break

    def move_mean_out(self, irr_range):
        i_pos, _ = self.componentIrreducibilitySampling()
        #pdb.set_trace()
        mu_pos = self.mu_pos[i_pos]
        i_neg, _ = self.closestNegComp(mu_pos)
        mu_neg = self.mu_neg[i_neg]
        delta = mu_pos - mu_neg
        self.mu_pos[i_pos] = self.mu_pos[i_pos] + 0.1 * delta
        metricsPN = self.computePNDataMetrics( )

        if metricsPN['irreducibility'] > irr_range[1]:
            k = 0
            lower = self.mu_pos[i_pos] - delta
            upper = self.mu_pos[i_pos]
            while ((metricsPN['irreducibility'] > irr_range[1]) or (metricsPN['irreducibility'] < irr_range[0])) and (k < 5):
                self.mu_pos[i_pos] =  (lower + upper)/2
                metricsPN = self.computePNDataMetrics( )
                if irr_range[0] <= metricsPN['irreducibility'] <= irr_range[1]:
                    break
                if metricsPN['irreducibility'] > irr_range[1]:
                    upper = self.mu_pos[i_pos]
                else:
                    lower = self.mu_pos[i_pos]
                k = k + 1

    def move_mean_in(self, irr_range):
        i_pos = randint(0, self.n_comps_pos - 1)
        upper = self.mu_pos[i_pos]
        i_neg, _ = self.closestNegComp(upper)
        lower = self.mu_neg[i_neg]
        metricsPN = self.computePNDataMetrics( )
        k = 0
        while (metricsPN['irreducibility'] > irr_range[1]) and (k < 5):
            self.mu_pos[i_pos] = (lower + upper)/2
            metricsPN = self.computePNDataMetrics( )
            if irr_range[0] <= metricsPN['irreducibility'] <= irr_range[1]:
                break
            if metricsPN['irreducibility'] > irr_range[1]:
                upper = self.mu_pos[i_pos]
            else:
                lower = self.mu_pos[i_pos]
            k = k + 1

    def align_covar(self, irr_range):
        i_pos = randint(0, self.n_comps_pos - 1)
        sig_pos = self.sig_pos[i_pos]
        i_neg, _ = self.closestNegComp(self.mu_pos[i_pos])
        sig_neg = self.sig_neg[i_neg]
        metricsPN = self.computePNDataMetrics( )
        k = 0
        up = 1.0
        low = 0.0
        while (metricsPN['irreducibility'] > irr_range[1]) and (k < 5):
            a = (up + low)/2.0
            self.sig_pos[i_pos] = a * sig_pos + (1-a) * sig_neg
            metricsPN = self.computePNDataMetrics( )
            if irr_range[0] <= metricsPN['irreducibility'] <= irr_range[1]:
                break
            if metricsPN['irreducibility'] > irr_range[1]:
                up = a
            else:
                low = a
            k = k + 1

    def equalMeans(self):
        epsilon = 10**-1
        ix_delta = [self.closestNegComp(mu) for mu in self.mu_pos]
        ix_delta = list(zip(*ix_delta))
        delta = ix_delta[1]
        sum = np.sum(delta, axis=0)
        return sum < epsilon

    def closestNegComp(self, mean):
        delta = np.array([np.sum((mean-mu)**2, axis=0) for mu in self.mu_neg])
        ix = np.argmin(delta, axis=0)
        return ix, delta[ix]

    def sigmaRatio(self, sigma):
        det = np.linalg.det(sigma)
        return np.array([det/np.linalg.det(sig_neg) for sig_neg in self.sig_neg])


    def componentIrreducibilitySampling(self):
        dg = self.createDataGenerator(500, 2000)
        comp_irr = [np.mean(dg.pn_posterior_balanced(comp.rvs(size=500)), axis=0) for comp in dg.components_pos]
        comp_irr = np.array(comp_irr)
        p = (1-comp_irr)/np.sum(1-comp_irr, axis=0)
        ix = np.random.choice(self.n_comps_pos, 1, p=p)
        ix = np.reshape(ix, newshape=())
        return ix, comp_irr




class NormalMixPNParameters:
# Used to gnerate PN data with desired AUC and irreducibility.
# The P and N distributions are mixtures of gaussians with equal number of components
    def __init__(self, dim, max_comps):
        self.dim = dim
        self.max_comps = max_comps
        #self.n_comps_pos = randint(1, max_comps)
        #self.n_comps_neg = randint(1, max_comps)
        self.n_comps_pos = max_comps
        self.n_comps_neg = max_comps
        self.mu_pos = list()
        self.mu_neg = list()
        for i in np.arange(max(self.n_comps_pos, self.n_comps_neg)):
            mu = np.array([2* random() - 1 for i in np.arange(self.dim)])
            mu = 0.5*mu/np.sqrt(self.dim)
            if i < self.n_comps_pos:
                self.mu_pos.append(mu)
            if i < self.n_comps_neg:
                self.mu_neg.append(mu)
        #self.mu_pos = [np.zeros(dim) for j in np.arange(self.n_comps_pos)]
        #self.mu_neg = [np.zeros(dim) for j in np.arange(self.n_comps_neg)]
        self.sig_pos = [np.identity(dim) for j in np.arange(self.n_comps_pos)]
        self.sig_neg = [np.identity(dim) for j in np.arange(self.n_comps_neg)]
        self.p_pos = dirichlet(np.ones(self.n_comps_pos)).rvs([])
        #self.p_neg = dirichlet(np.ones(self.n_comps_neg)).rvs([])
        self.p_neg = self.p_pos
        #self.changeInfo = {'changed': False, 'positive': True, 'mu': True, 'ix':0, 'oldvalue': self.mu_pos[0]}
        self.changeInfo = {'changed': False}
        self.alpha = random()
        self.updateDataGenerator()

    def computePNDataMetrics(self):
        epsilon = 0.05
        n = 10000
        x, y, _, x_p = self.dg.pn_data(n, 0.5)[0:4]
        posterior_pos = self.dg.pn_posterior_balanced(x_p)
        irreducibility = np.mean(np.cast['int32'](posterior_pos > 1-epsilon).flatten())
        posterior_x = self.dg.pn_posterior_balanced(x)
        fpr, tpr, thresholds = metrics.roc_curve(y, posterior_x)
        aucpn = metrics.auc(fpr, tpr)
        #pdb.set_trace()
        return {'aucpn': aucpn, 'irreducibility': irreducibility}

    def updateDataGenerator(self):
        self.dg = MVNormalMixDG(self.mu_pos, self.sig_pos, self.p_pos, self.mu_neg, self.sig_neg, self.p_neg, self.alpha)

    # def generatePNData(self, n_pos, n_neg):
    #     pos, c_pos = self.dg.data_pos(n_pos)
    #     neg, c_neg = self.dg.data_neg(n_neg)
    #     y = np.concatenate((np.ones([n_pos, 1]), np.zeros([n_neg, 1])), axis=0)
    #     x = np.concatenate((pos, neg), axis=0)
    #     xy = np.concatenate((x, y), axis=1)
    #     #pdb.set_trace()
    #     return xy, x, y, pos, neg, dg

    def perturb2SatisfyMetrics(self, irr_range, aucpn_range):
        irr_mid = np.mean(irr_range, axis=0)
        aucpn_min = min_aucpn(irr_mid)
        # if aucpn_range[0] < aucpn_min:
        #    raise ValueError('Irreducibility range and AUCPN range are not compatible:\n',
        #                      'AUCPN should be above', aucpn_min, 'for midpoint irreducibility of', irr_mid)
        while not self.isMetricSatisfied(irr_range, aucpn_range):
            self.markRandomParForChange()
            #print(self.changeInfo)
            if self.muMarked():
                self.perturbMu(irr_range, aucpn_range)
            else:
                if self.pMarked():
                    self.perturbProportion(irr_range, aucpn_range)
                else:
                    if random() <= 0.5:
                        self.perturbSigmaShape(irr_range, aucpn_range)
                    else:
                        self.perturbSigmaScale(irr_range, aucpn_range)
            self.commitChange()

    def perturbMu(self, irr_range, aucpn_range):
        print('Mu Perturb')
        c = 1
        delta = np.array([2 * random( ) - 1 for i in np.arange(self.dim)])
        delta = c * delta/np.linalg.norm(delta)
        mu = self.getMarkedParOldValue()
        up = 1.0
        self.proposeChange(mu + up * delta)
        while not self.isMetricUBSatisfied(irr_range, aucpn_range):
            up = up/2
            self.proposeChange(mu + up * delta)

    def perturbSigmaShape(self, irr_range, aucpn_range):
        print('Sigma Shape Perturb')
        newsigma = spd(self.dim)
        sigma = self.getMarkedParOldValue()
        a = 1.0
        self.proposeChange((1-a) * sigma + a * newsigma)
        while (not (self.isMetricUBSatisfied(irr_range, aucpn_range) and self.acceptableSigma((1-a) * sigma + a * newsigma))):
            a = a/2
            self.proposeChange((1-a) * sigma + a * newsigma)
        if not self.acceptableSigma((1-a) * sigma + a * newsigma):
            self.revert2OldValue()

    def perturbSigmaScale(self, irr_range, aucpn_range):
        print('Sigma Scale Perturb')
        sigma = self.getMarkedParOldValue()
        a = 0.5
        self.proposeChange(a * sigma)
        while (not (self.isMetricUBSatisfied(irr_range, aucpn_range) and self.acceptableSigma(a * sigma))):
            a = 1 + (a - 1)/2
            print(a)
            #print('metric:', self.isMetricUBSatisfied(irr_range, aucpn_range))
            #print('acceptable Sigma:', self.acceptableSigma(a * sigma) )
            self.proposeChange(a * sigma)
        if not self.acceptableSigma(a * sigma):
            self.revert2OldValue()

    def perturbProportion(self, irr_range, aucpn_range):
        print('Perturb Proportion')
        prop = self.getMarkedParOldValue( )
        a = 1
        if self.changeInfo['is_positive']:
            prop_1 = dirichlet(np.ones(self.n_comps_pos)).rvs([])
        else:
            prop_1 = dirichlet(np.ones(self.n_comps_neg)).rvs([])
        new_prop = (1 - a) * prop + a * prop_1
        self.proposeChange(new_prop)
        while not (self.isMetricUBSatisfied(irr_range, aucpn_range)):
            a = a/2
            new_prop = (1 - a) * prop + a * prop_1
            # print(a)
            self.proposeChange(new_prop)

    def muMarked(self):
        return self.changeInfo['is_mu']

    def pMarked(self):
        return self.changeInfo['is_proportion']

    def acceptableSigma(self, sigma):
        det = np.linalg.det(sigma)
        ratios = np.array([det/np.linalg.det(sig) for sig in self.sig_pos + self.sig_neg])
        print('Sigma Ratios: '+ str(ratios))
        #ratios[:] = 1
        return all(ratios >= 0.5) and all(ratios <= 2.0)

    def isMetricSatisfied(self, irr_range, aucpn_range):
        metrics = self.computePNDataMetrics()
        irr_satisfied = irr_range[0] <= metrics['irreducibility'] <= irr_range[1]
        auc_satisfied = aucpn_range[0] <= metrics['aucpn'] <= aucpn_range[1]
        print(metrics)
        return irr_satisfied and auc_satisfied

    def isMetricUBSatisfied(self, irr_range, aucpn_range):
        metrics = self.computePNDataMetrics()
        irr_satisfied = metrics['irreducibility'] <= irr_range[1]
        auc_satisfied = metrics['aucpn'] <= aucpn_range[1]
        return irr_satisfied and auc_satisfied

    def proposeChange(self, newValue):
        self.changeInfo['changed'] = True
        V = SN(**self.changeInfo)
        self.updatePar(V.is_positive, V.is_mu, V.is_proportion, V.ix, newValue)

    def commitChange(self):
        self.changeInfo = {'changed': False}

    def updatePar(self, is_positive, is_mu, is_proportion, ix, newValue):
        if is_positive:
            if is_mu:
                self.mu_pos[ix] = newValue
            else:
                if is_proportion:
                    self.p_pos = newValue
                else:
                    self.sig_pos[ix] = newValue
        else:
            if is_mu:
                self.mu_neg[ix] = newValue
            else:
                if is_proportion:
                    self.p_neg = newValue
                else:
                    self.sig_neg[ix] = newValue
        self.updateDataGenerator()

    def markRandomParForChange(self):
        if self.changeInfo['changed']:
            raise ValueError('Attempting to change a new parameter before committing the previous one')
        is_positive = random() < 0.5
        rr = random()
        is_mu = rr < 1.0/3.0
        is_proportion = 1.0/3.0 <= rr <= 2.0/3.0
        ix = np.nan
        if is_positive:
            ix = randint(0, self.n_comps_pos - 1)
            if is_mu:
                value = self.mu_pos[ix]
            else:
                if is_proportion:
                    value = self.p_pos
                else:
                    value = self.sig_pos[ix]
        else:
            ix = randint(0, self.n_comps_neg - 1)
            if is_mu:
                value = self.mu_neg[ix]
            else:
                if is_proportion:
                    value = self.p_neg
                else:
                    value = self.sig_neg[ix]

        self.changeInfo.update({'is_positive': is_positive, 'is_mu': is_mu, 'is_proportion': is_proportion, 'ix': ix, 'oldValue': value})

    def getMarkedParOldValue(self):
        return self.changeInfo['oldValue']

    def revert2OldValue(self):
        print('reverting to old value')
        V = SN(**self.changeInfo)
        self.updatePar(V.is_positive, V.is_mu, V.is_proportion, V.ix, V.oldValue)
        self.changeInfo['changed'] = False
        return



def min_aucpn(irreducibility):
    return irreducibility + (1-irreducibility)/2



class NormalMixPNParameters2:
# Used to generate PN data. The P and N distributions are mixtures of gaussians with equal number of components
# The positive and negative components are paired. Each pair of components satisfy the AUC and irreducibility constraints
    def __init__(self, dim, nComps):
        self.dim = dim
        self.nComps = nComps
        self.NormalPNPars = [NormalMixPNParameters(dim, 1) for i in np.arange(nComps)]
        self.alpha = random()
        self.p_pos = dirichlet(np.ones(self.nComps)).rvs([])
        self.p_neg = dirichlet(np.ones(self.nComps)).rvs([])
        self.updateDataGenerator()


    def perturb2SatisfyMetrics(self, irr_range, aucpn_range):
        [self.NormalPNPars[i].perturb2SatisfyMetrics(irr_range, aucpn_range) for i in np.arange(self.nComps)]
        #pdb.set_trace()
        for i in np.arange(self.nComps):
            pair1 = self.NormalPNPars[i]
            aucs = np.array([self.betweenPairAUC(i, j) for j in range(self.nComps) if j != i])
            #pdb.set_trace()
            while np.any(aucs < 0.7):
                #randomUnitVec = np.array([16/np.sqrt(self.dim) * random() - 8/np.sqrt(self.dim) for i in np.arange(self.dim)])
                randomUnitVec = np.array([2 * random() - 1 for i in np.arange(self.dim)])
                randomUnitVec = 0.5 * randomUnitVec / np.sqrt(self.dim)
                randomUnitVec = randomUnitVec/np.linalg.norm(randomUnitVec)
                pair1.mu_pos[0] = pair1.mu_pos[0] + randomUnitVec
                pair1.mu_neg[0] = pair1.mu_neg[0] + randomUnitVec
                aucs = np.array([self.betweenPairAUC(i, j) for j in range(self.nComps) if j != i])
                #pdb.set_trace()

        self.updateDataGenerator()

    def betweenAllPairsAUC(self):
        #cmb = combinations(np.arange(self.nComps), 2)
        aucs = [self.betweenPairAUC(i, j) for i, j in combinations(np.arange(self.nComps), 2)]
        #[print(str(i)+str(j)) for i, j in cmb]
        return aucs

    def betweenPairAUC(self, i, j):
        #pdb.set_trace()
        self.NormalPNPars[i].updateDataGenerator()
        self.NormalPNPars[j].updateDataGenerator()
        posi = self.NormalPNPars[i].dg.dist_pos
        negi = self.NormalPNPars[i].dg.dist_neg
        posj = self.NormalPNPars[j].dg.dist_pos
        negj = self.NormalPNPars[j].dg.dist_neg
        auc1 = AUCFromDistributions(posi, posj)
        auc2 = AUCFromDistributions(posi, negj)
        auc3 = AUCFromDistributions(negi, posj)
        auc4 = AUCFromDistributions(negi, negj)
        return [auc1, auc2, auc3, auc4]

    def withinPairAUC(self):
        #pdb.set_trace()
        [self.NormalPNPars[i].updateDataGenerator() for i in np.arange(self.nComps)]
        aucs = [AUCFromDistributions(pnPar.dg.dist_pos, pnPar.dg.dist_neg)  for pnPar in self.NormalPNPars]
        return aucs

    def updateDataGenerator(self):
        self.mu_pos = [self.NormalPNPars[i].mu_pos[0] for i in np.arange(self.nComps)]
        self.mu_neg = [self.NormalPNPars[i].mu_neg[0] for i in np.arange(self.nComps)]
        self.sig_pos = [self.NormalPNPars[i].sig_pos[0] for i in np.arange(self.nComps)]
        self.sig_neg = [self.NormalPNPars[i].sig_neg[0] for i in np.arange(self.nComps)]
        self.dg = MVNormalMixDG(self.mu_pos, self.sig_pos, self.p_pos, self.mu_neg, self.sig_neg, self.p_neg, self.alpha)



def AUCFromDistributions(dist1, dist2):
    dg = DataGenerator(dist1, dist2, 0.5)
    n=5000
    x, y = dg.pn_data(n)[0:2]
    posterior_x = dg.pn_posterior_balanced(x)
    fpr, tpr, thresholds = metrics.roc_curve(y, posterior_x)
    aucpn = metrics.auc(fpr, tpr)
    return aucpn