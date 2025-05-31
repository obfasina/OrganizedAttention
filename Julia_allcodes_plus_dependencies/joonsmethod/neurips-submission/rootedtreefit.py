import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import scipy.spatial.distance as ssd
import random
import time
import hccfit
import utils

class RootedTreeFit():
    
    def __init__(self, d, alt = False, rand = True, tol = 1e-5):
        self.d = d
        self.n = self.d.shape[0]
        self.alt = alt
        self.rand = rand
        self.tol = tol
        self.d_T = np.zeros((self.n, self.n))
        
    def fit_treeM(self, pivot_idx = 0, method = 'hcc'):
        di = self.d[pivot_idx]
        d_max = di.max()
        gp = np.tile(di, (self.n,1)) + np.tile(di.reshape(self.n,1), (1,self.n)) - self.d
        gp /= 2.
        d_U = d_max - gp
        np.fill_diagonal(d_U, 0)
        gp_T = np.zeros((self.n, self.n))
        if(method == 'hcc'):
            U = hccfit.HccLinkage(d_U)
            U.learn_UM()
            gp_T = di.max() - U.d_U
        elif(method == 'gromov'):
            distArray = ssd.squareform(d_U)
            Z = linkage(distArray, 'single')
            D_gr = utils.linkage_to_distance_matrix(Z)
            gp_T = di.max() - D_gr
        elif(method == 'complete'):
            distArray = ssd.squareform(d_U)
            Z = linkage(distArray, 'complete')
            D_cl = utils.linkage_to_distance_matrix(Z)
            gp_T = di.max() - D_cl
        elif(method == 'average'):
            distArray = ssd.squareform(d_U)
            Z = linkage(distArray, 'average')
            D_avg = utils.linkage_to_distance_matrix(Z)
            gp_T = di.max() - D_avg
        self.d_T = np.tile(di, (self.n,1)) + np.tile(di.reshape(self.n,1), (1,self.n)) - 2. * gp_T
        np.fill_diagonal(self.d_T, 0)