
import numpy as np
import matplotlib.pyplot as plt
import math
import sys, os


class tensor_paraproducts_haar:
    
    def __init__(self,f,Af,var):

        "var = string denoting nonlinearity"
        
        self.f = f
        self.Af = Af
        self.var = var

    def fderiv(self,avgmat):

        if self.var == 'softmax':
            out = np.exp(avgmat)
        if self.var == 'gaussian':
            # epsilon = 1
            out = -np.exp(avgmat)
        if self.var == 'potential':
            out = -(avgmat**(-2))
        
        return out

    def sderiv(self,avgmat):

        if self.var == 'softmax':
            out = np.exp(avgmat)
        if self.var == 'gaussian':
            # epsilon = 1
            out = np.exp(avgmat)
        if self.var == 'potential':
            out = 2*(avgmat**(-3))
        
        return out

    def compute_avgop(self,j,jp):

        finp = self.f
        Ny = finp.shape[0]
        Nx = finp.shape[1]
        dy=int(2**(-jp)*Ny);    
        dx=int(2**(-j)*Nx);
        
        onesmat = np.ones((dy,dx))
        avgop = np.zeros((Ny,Nx))
        
        for i in range(0,Nx,dx):
            for j in range(0,Ny,dy):
        
                xstart = i
                xend = i + dx
                ystart = j
                yend = j + dy
                avgop[ystart:yend,xstart:xend] = np.mean(finp[ystart:yend,xstart:xend]) * onesmat
                
        return avgop


    def paraproduct_decomp(self,xscales,yscales):

        finp = self.f
        Ny = finp.shape[0]
        Nx = finp.shape[1]
    
        tenswave = np.zeros((Ny,Nx))
        xwave = np.zeros((Ny,Nx))
        ywave = np.zeros((Ny,Nx))
        fapprox = np.zeros((Ny,Nx))
        scapprox = np.zeros((Ny,Nx))
        totapprox = np.zeros((Ny,Nx))
        
        
        for j_ in xscales:
            for jp_ in yscales:
    
        
                avgop_jxjy = self.compute_avgop(j_,jp_)
                avgop_jxpjy = self.compute_avgop(j_ + 1,jp_)
                avgop_jxjyp = self.compute_avgop(j_,jp_ + 1)
                avgop_jxpjyp = self.compute_avgop(j_ + 1,jp_ + 1)
        
                tenswave = tenswave + (avgop_jxjy - avgop_jxpjy - avgop_jxjyp + avgop_jxpjyp)
                xwave = xwave + (avgop_jxpjy - avgop_jxjy)
                ywave = ywave + (avgop_jxjyp - avgop_jxjy)
        
                ford = self.fderiv(avgop_jxjy)
                scord = self.sderiv(avgop_jxjy)
        
                fapprox = fapprox + (ford * tenswave) 
                scapprox = scapprox + (scord * xwave * ywave)
                curapprox = fapprox + scapprox
                totapprox = totapprox + curapprox
                
        resid = np.abs(totapprox - self.Af)
    
        decomp = {}
        decomp['first_ord'] = fapprox
        decomp['second_ord'] = scapprox
        decomp['total'] = totapprox
        decomp['resid'] = resid

        return decomp







        