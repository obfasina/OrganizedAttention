import sys
import os

# Add the directory containing Packages.py to the Python path
sys.path.append(os.path.abspath('./pyquest-master/'))
from imports import *
import questionnaire2 as q2coif
import numpy as np
import scipy.spatial as spa
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import Apply_questionnaire_to_data as quaest

a=10
b=5
rng=np.random.default_rng(1331)
Npoints=500
randomsel=np.sort(rng.random(Npoints))
X=0.05*a*np.cos(2*np.pi*randomsel)
Y=0.05*b*np.sin(2*np.pi*randomsel)

Z=np.stack((X,Y))
Dm=spa.distance.squareform(spa.distance.pdist(Z.T))
Kern_osc=np.sin(2*np.pi*3*Dm)
Kern_decay=np.nan_to_num(Kern_osc/Dm,neginf=0,posinf=0)+3*np.eye(Npoints)
plt.figure()
plt.imshow(Kern_osc)
plt.figure()
plt.imshow(Kern_decay)
Kern_main=Kern_osc
perm=rng.choice(Npoints,size=Npoints,replace=False)
perm_inv=np.argsort(perm)
Kern_disturbed=Kern_main[perm][:,perm]
fig,axs=plt.subplots(nrows=1,ncols=2)
axs[0].imshow(Kern_main)
axs[1].imshow(Kern_disturbed)
axs[0].set_title("Original Kernel")
axs[1].set_title("After Permutation")

tree_original_ord=quaest.main(Kern_main)
tree_perm_ord=quaest.main(Kern_disturbed)
tree_R1=tree_original_ord.row_trees[-1]
tree_C1=tree_original_ord.col_trees[-1]

tree_R2=tree_perm_ord.row_trees[-1]
tree_C2=tree_perm_ord.col_trees[-1]

plt.figure()
plot_tree(tree_C1)
plt.figure()
plot_tree(tree_C2)


tree_original=quaest.main(Kern_main,True)
tree_perm=quaest.main(Kern_disturbed,True)
tree_R3=tree_original.row_trees[-1]
tree_C3=tree_original.col_trees[-1]

tree_R4=tree_perm.row_trees[-1]
tree_C4=tree_perm.col_trees[-1]

plt.figure()
plot_tree(tree_C3)
plt.figure()
plot_tree(tree_C4)


def get_order(tree_quest):
    tree_order = [x.elements[0] for x in tree_quest.dfs_leaves()]
    return tree_order

order_C1=get_order(tree_C1)
order_R1=get_order(tree_R1)

order_C2=get_order(tree_C2)
order_R2=get_order(tree_R2)


fig,axs=plt.subplots(nrows=1,ncols=2)
axs[0].imshow(Kern_main[order_R1][:,order_C1])
axs[1].imshow(Kern_disturbed[order_R2][:,order_C2])
axs[0].set_title("Original Kernel after questionnaire")
axs[1].set_title("Permuted Kernel after questionnaire")
#fig.suptitle("After tree ordering")

order_C3=get_order(tree_C3)
order_R3=get_order(tree_R3)

order_C4=get_order(tree_C4)
order_R4=get_order(tree_R4)

fig,axs=plt.subplots(nrows=1,ncols=2)
axs[0].imshow(Kern_main[order_R3][:,order_C3])
axs[1].imshow(Kern_disturbed[order_R4][:,order_C4])
axs[0].set_title("Original Kernel after questionnaire")
axs[1].set_title("Permuted Kernel after questionnaire")
#fig.suptitle("Before tree ordering")
treedist_1=tree_C1.tree_distance_mat()
treedist_2=tree_C2.tree_distance_mat()


H=np.eye(Npoints)-(1/Npoints)*np.outer(np.ones(Npoints),np.ones(Npoints))
GramTree=-(1/2)*np.matmul(H,np.matmul(np.multiply(treedist_1,treedist_1),H))
def diffmap_median(A,alpha):
    med=np.median(A)/2
    K1=np.exp(-np.multiply(A/med,A/med))
    d1=np.power(np.sum(K1,axis=1),alpha)
    K2=np.multiply(1/d1,np.multiply(1/d1,K1.T).T)
    dhalfinv=np.power(np.sum(K2,axis=1),-1/2)
    Ksym=np.multiply(dhalfinv,np.multiply(dhalfinv,K2.T).T)
    return Ksym,dhalfinv

plt.figure()
plt.imshow(GramTree)
S,U=np.linalg.eigh(GramTree)
S=S[::-1]
U=U[np.argsort(S)[::-1]]
plt.figure()
plt.imshow(treedist_1)
plt.figure()
plt.plot(U[:,:4])
Dmap_sym,dhalfinv=diffmap_median(Dm,1)
S2,U2=np.linalg.eigh(Dmap_sym)
plt.figure()
plt.plot(U2[:,:4])
plt.figure()
plt.imshow(Dm)
plt.show()

print('wat')