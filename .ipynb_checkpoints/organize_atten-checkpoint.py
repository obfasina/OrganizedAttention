import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.io import savemat
import sys,os
sys.path.append("./Julia_allcodes_plus_dependencies/pyquest-master")
import diffusion_maps
from imports import *


class organize_heads:

    def __init__(self,X):

        "X = tensor to be analyzed"
        self.X = X

        return


    def init_quest_params(self,ralph,rbeta,calph,cbeta,dalph,dbeta,niters):

        init_aff = questionnaire.INIT_AFF_COS_SIM
        row_tree_type = questionnaire.TREE_TYPE_FLEXIBLE
        col_tree_type = questionnaire.TREE_TYPE_FLEXIBLE
        chan_tree_type = questionnaire.TREE_TYPE_FLEXIBLE
        
        row_aff_type = questionnaire.DUAL_EMD
        col_aff_type = questionnaire.DUAL_EMD
        chan_aff_type = questionnaire.DUAL_EMD
        
        kwargs = {}
        kwargs["n_iters"] = niters
        kwargs["row_beta"] = rbeta
        kwargs["col_beta"] = cbeta
        kwargs["chan_beta"] = dbeta
        kwargs["row_alpha"] = ralph
        kwargs["col_alpha"] = calph
        kwargs["chan_alpha"] = dalph
        kwargs["row_tree_constant"] = 1
        
        init_tree = [row_tree_type,col_tree_type,chan_tree_type]
        params = questionnaire.PyQuest3DParams(init_aff,init_tree,
                     row_aff_type,col_aff_type,chan_aff_type,**kwargs)

        self.params = questionnaire.PyQuest3DParams(init_aff,init_tree,
             row_aff_type,col_aff_type,chan_aff_type,**kwargs)

        return

    def run_quest(self):

        "Run questionnaire, save affinity and trees"

        self.qrun = questionnaire.pyquest3d(self.X,self.params)
        self.chan_aff = self.qrun.chan_aff
        self.chan_trees = self.qrun.chan_trees[-1]
        self.row_trees = self.qrun.row_trees[-1]
        self.col_trees = self.qrun.col_trees[-1]

        return


    def run_diffusion(self):

        "Run Diffusion map on kernel of attention heads (i.e. Channel Affinity) "

        init_diff = diffusion_maps.diffusion_maps(self.X)
        init_diff.K = self.chan_aff
        init_diff.compute_diffop()
        self.firstvec = init_diff.eivecs[:,0]
        self.secondvec = init_diff.eivecs[:,1]
        self.thirdvec = init_diff.eivecs[:,2]

        return

    def trihaar_proc(self,nqb,nkb):

        "Signal processing in trihaar basis"
        "nbq = number of query basis vectors"
        "nbk = number of key basis vectors"

        # Generate basis vectors along each axis
        self.QB = haar.compute_haar(self.row_trees)
        self.KB = haar.compute_haar(self.col_trees)
        self.CB = haar.compute_haar(self.chan_trees)

        # Generate tensor bases
        self.QKbases = []
        for j in range(nqb):
            for s in range(nkb):
                self.QKbases.append(np.outer(self.QB[:,j],self.KB[:,s]))

        # Sort in ascending order by nonzero support size      
        nonzro_supp = [np.count_nonzero(x) for x in self.QKbases]
        self.suppidx = np.flip(np.argsort(nonzro_supp))


        return


    def compute_ntwk_entp(self,k):

        "Generate top k expansion coefficients when projecting tensor into query,key,head basis"
        
        # Initialize query,key,head basis vectors
        self.trihaar_proc(0,0) 
        nqb = self.QB.shape[1]
        nkb = self.KB.shape[1]
        nhb = self.CB.shape[1]
        
        # precompute top k query-key basis vectors
        qksupps = []
        qkidxs = []
        for i in range(nqb):
            for j in range(nkb):
                outprod = np.outer(self.QB[:,i],self.KB[:,j]).flatten()
                nzroelem = [x for x in outprod if x > 1e-4]
                supp = len(nzroelem)
                qksupps.append(supp)
                qkidxs.append((i,j))
                
        
        # top k  query-key tensor basis vectors (by support size)
        srt_qksupps = np.argsort(qksupps)
        qkidxs = np.array(qkidxs)
        tp_qkidxs = np.flip(qkidxs[srt_qksupps[-k:]])
        
        
        hqksupps = []
        hqkidxs = []
        qkbases = []
        for m in range(nhb):
            c = 0
            for x in tp_qkidxs:
                qktb = np.outer(self.QB[:,x[0]],self.KB[:,x[1]])
                hqktb = np.einsum('jk,i->jki',qktb,self.CB[:,m])
                nzroelem = [x for x in outprod if x > 1e-4]
                supp = len(nzroelem)
                hqksupps.append(supp)
                hqkidxs.append((m,c))
                qkbases.append(qktb)
                c+=1  
        
        
        srt_hqksupps = np.argsort(hqksupps)
        hqkidxs = np.array(hqkidxs)
        tp_hqkidxs = np.flip(hqkidxs[srt_hqksupps[-k:]])
        
        expan = []
        for x in tp_hqkidxs:
            hidx,qkbidx = x[0],x[1]
            qktb = qkbases[qkbidx]
            hqktb = np.einsum('jk,i->jki',qktb,self.CB[:,hidx])
            expan.append(np.abs(np.sum(np.sum(self.X*hqktb))))
    
        return expan

        


    