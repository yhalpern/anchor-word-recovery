from __future__ import division
from sys import stdout
import numpy as np
import time
import scipy.sparse as sparse
import math
from multiprocessing import Pool
from  helper_functions import *

# Given a sparse CSC document matrix M (with floating point entries),
# comptues the word-word correlation matrix Q
def generate_Q_matrix(row_M, col_M, row_normalize=False, projection_matrix=None, indices=None):
    simulation_start = time.time()
    vocabSize = row_M.shape[0]
    numdocs = row_M.shape[1]
    diag_M = np.zeros((vocabSize, 1))
    columns = []
    for j in xrange(numdocs):
        if j % 1000 == 0:
            stdout.write('\r normalizing'+str(j)+' / '+str(numdocs))
            stdout.flush()
        s = time.time()
        wpd = col_M[:, j].sum()
        col = col_M[:, j].multiply(1./(wpd*(wpd-1)))
        columns.append(col)
        for i, idx in enumerate(col.indices):
            diag_M[idx,0] += col.data[i]
        
    col_M = sparse.hstack(columns)

    print 'done normalizing'
    print 'vocab', vocabSize
        
    def generate_vector(i):
        Q_vec = row_M[i, :] * col_M.T
        Q_vec[0,i] -= diag_M[i]
        Q_vec.data /= float(numdocs)
        if row_normalize:
            Q_vec.data /= float(Q_vec.sum())
        if projection_matrix is not None:
            Q_vec = Q_vec.dot(projection_matrix)
        else:
            Q_vec = np.array(Q_vec.todense()).reshape(vocabSize)
        return Q_vec

    if indices is None:
        args = xrange(vocabSize)
    else:
        args = indices

    for i,Q_vec in enumerate((generate_vector(arg) for arg in args)):
        if i % 1 == 0:
            stdout.write('\r generating row '+str(i)+' / '+str(vocabSize))
            stdout.flush()
        yield Q_vec
    print 'done'
