import sys
import os

# Add the directory containing Packages.py to the Python path
sys.path.append(os.path.abspath('./pyquest-master/'))
from imports import *
import questionnaire2 as q2coif
import questionnaire3 as q3coif



def main(A,switchver=False):

    ### Now for Raphy's questionnaire method
    switchcos = 0
    kwargs = {}
    kwargs["threshold"] = 0.0
    kwargs["row_alpha"] = 0.0
    kwargs["col_alpha"] = 0.0
    kwargs["row_beta"] = 1.0
    kwargs["col_beta"] = 1.0
    kwargs["tree_constant"] = 1.0
    kwargs["n_iters"] = 6

    if switchcos > 0:
        init_row_aff = affinity.mutual_cosine_similarity(A.T, threshold=0.0)
        init_col_aff = affinity.mutual_cosine_similarity(A, threshold=0.0)
        params = q2coif.PyQuestParams(q2coif.INIT_AFF_COS_SIM,
                                      q2coif.TREE_TYPE_FLEXIBLE,
                                      q2coif.DUAL_EMD,
                                      q2coif.DUAL_EMD, **kwargs)
    else:
        init_row_aff = affinity.gaussian_euclidean(A.T)
        init_col_aff = affinity.gaussian_euclidean(A)
        params = q2coif.PyQuestParams(q2coif.INIT_AFF_GAUSSIAN,
                                      q2coif.TREE_TYPE_FLEXIBLE,
                                      q2coif.DUAL_EMD,
                                      q2coif.DUAL_EMD, **kwargs)

    init_row_vecs, init_row_vals = markov.markov_eigs(init_row_aff, 12)
    init_col_vecs, init_col_vals = markov.markov_eigs(init_col_aff, 12)
    if switchver:
        qrun = q2coif.pyquest(A, params)
    else:
        qrun = q3coif.pyquest(A, params)
    return qrun
