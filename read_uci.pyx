import numpy as np
import sys
cimport numpy as np
import scipy

DTYPE = np.int
ctypedef np.int_t DTYPE_t


def readfile(infile, int nnz, int num_docs, int num_words):
    cdef np.ndarray[DTYPE_t] rows = np.zeros(nnz, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t] cols= np.zeros(nnz, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t] values= np.zeros(nnz, dtype=DTYPE)
    print nnz, type(nnz)
    
    cdef long idx = 0
    for idx in range(0, nnz):
        if idx % 10000 == 0:
            sys.stdout.write('\r'+str(idx)+'/'+str(nnz))
        col, row, value = infile.readline().split()    
        rows[idx] = int(row)-1
        cols[idx] = int(col)-1
        values[idx] = int(value)

    return scipy.sparse.coo_matrix((values, (rows, cols)), shape=(num_words,num_docs))
