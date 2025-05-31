import numpy as np
import networkx as nx
from scipy.cluster.hierarchy import dendrogram, linkage
import random
import pickle
import os
import time

def linkage_to_distance_matrix(Z):
    N = Z.shape[0] + 1
    C = []
    for i in range(N):
        C.append([i])
    D = np.zeros((N,N))
    for i in range(N-1):
        j = Z[i,0].astype(int)
        k = Z[i,1].astype(int)
        for x in C[j]:
            for y in C[k]:
                D[x,y] = Z[i,2]
                D[y,x] = Z[i,2]
        C.append(C[j] + C[k])
    return D

def construct_rooted_tree(r, di, Z):
    dmax = Z[-1][2]
    G = nx.Graph()
    n = di.shape[0]
    dr = np.zeros(2*n-1)
    for i in range(n):
        dr[i] = di[i]
    G.add_nodes_from(range(2*n-1))
    for i in range(n-1):
        x = int(Z[i][0])
        y = int(Z[i][1])
        gp = dmax - Z[i][2]
        dr[i+n] = gp
        G.add_edge(i+n, x, weight = dr[x] - gp)
        G.add_edge(i+n, y, weight = dr[y] - gp)
        if(dr[x] - gp < 0):
            print('negative edge detected', x, i+n)
        if(dr[y] - gp < 0):
            print('negative edge detected', y, i+n)
    return G

# added for synthetic dataset

def all_triplets(L):
    triples = []
    N = len(L)
    for i in range(N):
        for j in range(i+1,N):
            for k in range(j+1,N):
                triples.append((L[i],L[j],L[k]))
    return triples

def all_pairs(L):
    pairs = []
    N = len(L)
    for i in range(N):
        for j in range(i+1,N):
            pairs.append((L[i], L[j]))
    return pairs

def gp(d,x,y,r):
    return 0.5*(d[x,r] + d[y,r] - d[x,y])

def fp(d,x,y,z,r):
    A = gp(d,x,y,r)
    B = gp(d,y,z,r)
    C = gp(d,x,z,r)
    max_gp = max(A,B,C)
    min_gp = min(A,B,C)
    return (A+B+C) - max_gp - 2 * min_gp

def hyp_vector(d, r = 0):
    Delta = {}
    N = d.shape[0]
    for i in range(N):
        for j in range(i+1,N):
            for k in range(j+1,N):
                Delta[(i,j,k)] = fp(d,i,j,k,r)
    return Delta

def NeighborJoin(d):
    n = d.shape[0]
    D = np.array(d, dtype = float)
    vertices = [i for i in range(n)]
    tree = nx.Graph()
    tree.add_nodes_from(range(n))
    if len(D) <= 1:
        return tree
    u = n
    while True:
        if n == 2:
            tree.add_edge(vertices[0], vertices[1], weight = D[0][1])
            break
        totalDist = np.sum(D, axis = 0)
        D1 = (n-2) * D
        D1 = D1 - totalDist
        D1 = D1 - totalDist.reshape((n, 1))
        MM= D1.max()
        np.fill_diagonal(D1, MM+1) # to ensure diagonal element should never be picked
#         print("Q = ", D1)
        index = np.argmin(D1)
        i = index // n
        j = index % n
        delta = (totalDist[i] - totalDist[j])/(n-2)
        li = (D[i, j]+delta)/2
        lj = (D[i, j]-delta)/2
        d_new = (D[i, :]+D[j, :]-D[i, j])/2
        D = np.insert(D, n, d_new, axis = 0)
        d_new = np.insert(d_new, n, 0., axis = 0)
        D = np.insert(D, n, d_new, axis = 1)
        D = np.delete(D, [i, j], 0)
        D = np.delete(D, [i, j], 1)
#         print("d(u, i) = ", li, "d(u,j) = ", lj)
        tree.add_node(u)
        vi = vertices[i]
        vj = vertices[j]
        tree.add_edge(u, vi, weight = li)
        tree.add_edge(u, vj, weight = lj)
#         print("vi, vj = ", vi, vj)
#         print("D = ", D)
        vertices.remove(vi)
        vertices.remove(vj)
        vertices.append(u)
        u += 1
        n -= 1

    return tree