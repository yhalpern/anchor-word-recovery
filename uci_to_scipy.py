#takes in matrix in UCI repository format and outputs a scipy sparse matrix file

import numpy 
import pyximport
pyximport.install()
import read_uci
import sys
import scipy.io

if len(sys.argv) < 2:
    print "usage: input_matrix output_matrix"
    sys.exit()

input_matrix = sys.argv[1]
output_matrix_name = sys.argv[2]

infile = file(input_matrix)
num_docs = int(infile.readline())
num_words = int(infile.readline())
nnz = int(infile.readline())
output_matrix = read_uci.readfile(infile, nnz, num_docs, num_words)
scipy.io.savemat(output_matrix_name, {'M' : output_matrix}, oned_as='column')
