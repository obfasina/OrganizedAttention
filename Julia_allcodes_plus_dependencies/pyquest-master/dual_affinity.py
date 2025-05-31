"""
dual_affinity.py: Functions for calculating dual affinity based on Earth 
                  Mover's Distance.
"""

import numpy as np
import tree_util
import scipy.spatial as spsp
import collections
import transform

def emd_dual_aff(emd,eps=1.0):
    """
    Calculates the EMD affinity from a distance matrix
    by normalizing by the median EMD and taking exp^(-EMD)
    without thresholding.
    """
   
    epall = eps*np.median(emd)
    if epall == 0.0:
        epall = 1.0
    
    return np.exp(-emd/epall)

def calc_emd(data,row_tree,alpha=1.0,beta=0.0,exc_sing=False,weights=None):
    """
    Calculates the EMD on the *columns* from data and a tree on the rows.
    each level is weighted by 2**((1-level)*alpha)
    each folder size (fraction) is raised to the beta power for weighting.
    """
    rows,_ = np.shape(data)
    assert rows == row_tree.size, "Tree size must match # rows in data."

    folder_fraction = np.array([((node.size*1.0/rows)**beta)*
                                (2.0**((1.0-node.level)*alpha))
                                 for node in row_tree])
    if weights is not None:
        folder_fraction = folder_fraction*weights
    
    if exc_sing:
        for node in row_tree:
            if node.size == 1:
                folder_fraction[node.idx] = 0.0
    coefs = tree_util.tree_averages(data,row_tree)
    
    ext_vecs = np.diag(folder_fraction).dot(coefs)
    
    pds = spsp.distance.pdist(ext_vecs.T,"cityblock")
    distances = spsp.distance.squareform(pds)

    return distances

def calc_emd_multi_tree(data,row_trees,alpha=1.0,beta=0.0,exc_sing=False):
    rows,cols = np.shape(data)

    ext_vecs = np.array([]).reshape(0,cols)
    
    n_trees = len(row_trees)
    
    for i in range(ntrees):
        row_tree = row_trees[i]
        assert rows == row_tree.size, "Tree size must match # rows in data."

        folder_fraction = np.array([((node.size*1.0/rows)**beta)*
                                    (2.0**((1.0-node.level)*alpha))
                                     for node in row_tree])
        if exc_sing:
            for node in row_tree:
                if node.size == 1:
                    folder_fraction[node.idx] = 0.0
        coefs = transform.averaging(data,row_tree)
        ext_vecs = np.vstack([ext_vecs, np.diag(folder_fraction).dot(coefs)]) 

    pds = spsp.distance.pdist(ext_vecs.T,"cityblock")
    distances = spsp.distance.squareform(pds)
    
    return distances / float(n_trees)
    
def calc_emd_multi_tree_ref(ref_data,data,row_trees,alpha=1.0,beta=0.0,exc_sing=False):
    rows,cols = np.shape(data)
    ref_rows,ref_cols = np.shape(ref_data)
    
    emd = np.zeros([ref_cols,cols])
    ntrees = len(row_trees)
    
    for i in range(ntrees):
        row_tree = row_trees[i]
        emd += calc_emd_ref(ref_data,data,row_tree,alpha=alpha,beta=beta)
    
    return emd/ float(ntrees)

    
def calc_emd_ref(ref_data,data,row_tree,alpha=1.0,beta=0.0):
    """
    Calculates the EMD from a set of points to a reference set of points
    The columns of ref_data are each a reference set point.
    The columns of data are each a point outside the reference set.
    """
    ref_rows,ref_cols = np.shape(ref_data)
    rows,cols = np.shape(data)
    assert rows == row_tree.size, "Tree size must match # rows in data."
    assert ref_rows == rows, "Mismatched row #: reference and sample sets."

    emd = np.zeros([ref_cols,cols])
    ref_coefs = tree_util.tree_averages(ref_data, row_tree)
    coefs = tree_util.tree_averages(data, row_tree)
    level_elements = collections.defaultdict(list)
    level_sizes = collections.defaultdict(int)
    
    for node in row_tree:
        level_elements[node.level].append(node.idx)
        level_sizes[node.level] += node.size
        
    folder_fraction = np.array([node.size for node in row_tree],np.float)
    for level in range(1,row_tree.tree_depth+1):
        fsize = np.sum(folder_fraction[level_elements[level]])
        folder_fraction[level_elements[level]] /= fsize
    
    folder_fraction = folder_fraction**beta
    coefs = np.diag(folder_fraction).dot(coefs)
    ref_coefs = np.diag(folder_fraction).dot(ref_coefs)
    for level in range(1,row_tree.tree_depth+1):
        distances = spsp.distance.cdist(coefs[level_elements[level],:].T,
                                        ref_coefs[level_elements[level],:].T,
                                        "cityblock").T
        emd += (2**((1.0-level)*alpha))*distances

    return emd
    
def calc_emd_ref2(ref_data,data,row_tree,alpha=1.0,beta=0.0,weights=None):
    """
    Calculates the EMD from a set of points to a reference set of points
    The columns of ref_data are each a reference set point.
    The columns of data are each a point outside the reference set.
    """
    ref_rows,ref_cols = np.shape(ref_data)
    rows,cols = np.shape(data)
    assert rows == row_tree.size, "Tree size must match # rows in data."
    assert ref_rows == rows, "Mismatched row #: reference and sample sets."

    emd = np.zeros([ref_cols,cols])
    
    averages_mat = transform.tree_averages_mat(row_tree)
    ref_coefs = averages_mat.dot(ref_data)
    coefs = averages_mat.dot(data)
    
    folder_fraction = np.array([((node.size*1.0/rows)**beta)*
                                (2.0**((1.0-node.level)*alpha))
                                 for node in row_tree])
    if weights is not None:
        folder_fraction = folder_fraction*weights
    
    coefs = np.diag(folder_fraction).dot(coefs)    
    ref_coefs = np.diag(folder_fraction).dot(ref_coefs)   
    
    emd = spsp.distance.cdist(ref_coefs.T,coefs.T,"cityblock")
    return emd

def calc_2demd(data,row_tree, col_tree, row_alpha=1.0, row_beta=0.0, 
	col_alpha=1.0, col_beta=0.0, exc_sing=False, exc_raw=False):
    """
    Calculates 2D EMD on database of data using a tree on the rows and columns.
    each level is weighted by 2**((1-level)*alpha)
    each folder size (fraction) is raised to the beta power for weighting.
    """
    nrows,ncols,nchannels = np.shape(data)
    assert nrows == row_tree.size, "Tree size must match # rows in data."
    assert ncols == col_tree.size, "Tree size must match # cols in data."
    
    row_folder_fraction = np.array([((node.size*1.0/nrows)**row_beta)*
                                (2.0**((1.0-node.level)*row_alpha))
                                 for node in row_tree])
    col_folder_fraction = np.array([((node.size*1.0/ncols)**col_beta)*
                                (2.0**((1.0-node.level)*col_alpha))
                                 for node in col_tree])
    if exc_sing:
        for node in row_tree:
            if node.size == 1:
                row_folder_fraction[node.idx] = 0.0
        for node in col_tree:
            if node.size == 1:
                col_folder_fraction[node.idx] = 0.0
    folder_frac = np.outer(row_folder_fraction, col_folder_fraction)
                      
    avgs = tree_util.bitree_averages(data[:,:,0], row_tree, col_tree)
    avgs = folder_frac * avgs
    
    if exc_raw:
        col_singletons_start = col_tree.tree_size - ncols
        row_singletons_start = row_tree.tree_size - nrows
        avgs = avgs[:row_singletons_start,:col_singletons_start]
    
    sums3d = np.zeros((nchannels,np.size(avgs)))
    
    sums3d[0,:] = np.reshape(avgs,(1,-1))
    for t in range(1,nchannels):
        avgs = tree_util.bitree_averages(data[:,:,t], row_tree, col_tree)
        avgs = folder_frac * avgs
        if exc_raw:
            avgs = avgs[:row_singletons_start,:col_singletons_start]
        sums3d[t,:] = np.reshape(avgs,(1,-1))
    
    pds = spsp.distance.pdist(sums3d, "cityblock")
    distances = spsp.distance.squareform(pds)

    return distances

def calc_2demd_ref(ref_data,data,row_tree,col_tree, row_alpha=1.0, row_beta=0.0, 
	col_alpha=1.0, col_beta=0.0, exc_sing=False,exc_raw=False):
    """
    Calculates the EMD from a set of points to a reference set of points
    The columns of ref_data are each a reference set point.
    The columns of data are each a point outside the reference set.
    """
    if data.ndim == 2:
        ref_rows,ref_cols = np.shape(ref_data)
        rows,cols = np.shape(data)
    else:
        ref_rows,ref_cols,ref_chans = np.shape(ref_data)
        rows,cols,chans = np.shape(data)

    col_singletons_start = col_tree.tree_size - cols
    row_singletons_start = row_tree.tree_size - rows
            
    assert rows == row_tree.size, "Tree size must match # rows in data."
    assert ref_rows == rows, "Mismatched row #: reference and sample sets."
    assert cols == col_tree.size, "Tree size must match # cols in data."
    assert ref_cols == cols, "Mismatched col #: reference and sample sets."

    row_folder_fraction = np.array([((node.size*1.0/rows)**row_beta)*
                                (2.0**((1.0-node.level)*row_alpha))
                                 for node in row_tree])
    col_folder_fraction = np.array([((node.size*1.0/cols)**col_beta)*
                                (2.0**((1.0-node.level)*col_alpha))
                                 for node in col_tree])
    if exc_sing:
        for node in row_tree:
            if node.size == 1:
                row_folder_fraction[node.idx] = 0.0
        for node in col_tree:
            if node.size == 1:
                col_folder_fraction[node.idx] = 0.0
    folder_frac = np.outer(row_folder_fraction, col_folder_fraction)
 
    if data.ndim == 2:
        ref_coefs = tree_util.bitree_averages(ref_data, row_tree, col_tree)
        coefs = tree_util.bitree_averages(data, row_tree, col_tree)
        coefs = folder_frac * coefs
        ref_coefs = folder_frac * ref_coefs
        
        if exc_raw:
            avgs = avgs[:row_singletons_start,:col_singletons_start]
        
        return spsp.distance.cityblock(coefs.flatten(),ref_coefs.flatten())
    else:
        if exc_raw:
            folder_frac = folder_frac[:row_singletons_start,:col_singletons_start] 
               
        sums3d = np.zeros((chans,np.size(folder_frac)))
        for t in range(0,chans):
            avgs = tree_util.bitree_averages(data[:,:,t], row_tree, col_tree)
            if exc_raw:
                avgs = avgs[:row_singletons_start,:col_singletons_start]
            avgs = folder_frac * avgs
            
            sums3d[t,:] = np.reshape(avgs,(1,-1))
        
        ref_sums3d = np.zeros((ref_chans,np.size(folder_frac)))
        for t in range(0,ref_chans):
            avgs = tree_util.bitree_averages(ref_data[:,:,t], row_tree, col_tree)
            if exc_raw:
                avgs = avgs[:row_singletons_start,:col_singletons_start]
            avgs = folder_frac * avgs
            
            ref_sums3d[t,:] = np.reshape(avgs,(1,-1))
          
        return spsp.distance.cdist(sums3d,ref_sums3d, "cityblock")       
    
