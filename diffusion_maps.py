
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance_matrix
from scipy.linalg import fractional_matrix_power


class diffusion_maps:
    
    def __init__(self,X):

        # X = [N,D] is the input data matrix. The rows are the attention heads, the columns are the flattened matrices.
        # N = the number of attention heads
        # D = the feature space of each attention head
        
        self.X = X

    def build_kernel(self,pk=2,alfak=2,knn=5):

        # NOTE: We use the kernel from the PHATE paramters
        
        # pk = Type of norm. p = 2 for Euclidean. 
        # alfak = decay rate parameter
        # knn = number of neighbors to consider for epsilon

        # Construct kernel
        self.A = distance_matrix(self.X,self.X,pk)
        self.K = np.zeros((self.A.shape[0],self.A.shape[1]))
        eps_k = self.A[:,knn]
        for i in range(self.A.shape[0]):
            for j in range(self.A.shape[1]):
                val = (0.5*np.exp(-((self.A[i,j]/eps_k[i])**alfak))) + (0.5*np.exp(-((self.A[i,j]/eps_k[j])**alfak))) 
                self.K[i,j] = val

        # Check for nan's
        for i in range(self.K.shape[0]):
            for j in range(self.K.shape[1]):
                if np.isnan(self.K[i,j]):
                    if i==j: # If nan is on diagonal, replace with 1 since there is 0 distance; otherwise, replace with 0
                        self.K[i,j] = 1
                    else:
                        self.K[i,j] = 0
                
        return

    def compute_diffop(self):

        # Construct symmetricized Markov matrix to build diffusion operator
        D = np.diag(np.sum(self.K,axis=1))
        Dmhlf = fractional_matrix_power(D, -0.5)
        self.M = Dmhlf @ self.K @ Dmhlf
        self.eivals, self.eivecs = np.linalg.eigh(self.M)
        
        return

    def compute_outliers(self,nstd=1):

        # Compute outliers from new diffusion map coordinates

        xe = np.expand_dims(self.eivecs[:,0],axis=1)
        ye = np.expand_dims(self.eivecs[:,1],axis=1)
        ze = np.expand_dims(self.eivecs[:,2],axis=1)
        
        
        posmat = np.concatenate((xe,ye,ze),axis=1)
        posavg = np.mean(posmat,axis=0)
        posavgl2 = np.linalg.norm(posavg[0]**2 + posavg[1]**2 + posavg[2]**2)
        postd = np.std(posmat,axis=0)
        postdl2 = np.linalg.norm(postd[0]**2 + postd[1]**2 + postd[2]**2)
        
        
        upr = posavgl2 + (postdl2*nstd)
        lowr = posavgl2 - (postdl2*nstd)
        self.outidx = []
        for i in range(posmat.shape[0]):
            norm =  np.linalg.norm(posmat[i,0]**2 + posmat[i,1]**2 + posmat[i,2]**2)
            if not lowr < norm < upr:
                self.outidx.append(i)

        return

    