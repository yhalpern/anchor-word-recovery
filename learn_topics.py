import sys
import random_projection as rp
import gram_schmidt_stable as gs
from numpy.random import RandomState
import numpy as np
from fastRecover import nonNegativeRecover#do_recovery
from anchors import findAnchors
import scipy.sparse as sparse
import time
from Q_matrix import generate_Q_matrix 
import scipy.io

class Params:

    def __init__(self, filename):
        self.log_prefix=None
        self.checkpoint_prefix=None
        self.seed = int(time.time())

        for l in file(filename):
            if l == "\n" or l[0] == "#":
                continue
            l = l.strip()
            l = l.split('=')
            if l[0] == "log_prefix":
                self.log_prefix = l[1]
            elif l[0] == "max_threads":
                self.max_threads = int(l[1])
            elif l[0] == "eps":
                self.eps = float(l[1])
            elif l[0] == "checkpoint_prefix":
                self.checkpoint_prefix = l[1]
            elif l[0] == "new_dim":
                self.new_dim = int(l[1])
            elif l[0] == "seed":
                self.seed = int(l[1])
            elif l[0] == "anchor_thresh":
                self.anchor_thresh = int(l[1])
            elif l[0] == "top_words":
                self.top_words = int(l[1])


if __name__ == "__main__":
    #parse input args
    if len(sys.argv) > 6:
        infile = sys.argv[1]
        settings_file = sys.argv[2]
        vocab_file = sys.argv[3]
        K = int(sys.argv[4])
        loss = sys.argv[5]
        outfile = sys.argv[6]

    else:
        print "usage: ./learn_topics.py word_doc_matrix settings_file vocab_file K loss output_filename"
        print "for more info see readme.txt"
        sys.exit()

    params = Params(settings_file)
    params.dictionary_file = vocab_file
    M = scipy.io.loadmat(infile)['M']
    col_M = sparse.csc_matrix(M)
    row_M = sparse.csr_matrix(M)


    vocab = file(vocab_file).read().strip().split()
    V = M.shape[0]
    prng = RandomState(params.seed)
    R = rp.Random_Matrix(V, params.new_dim, prng)
    #Q = np.vstack(generate_Q_matrix(row_M, col_M, row_normalize=True, projection_matrix=R.T)) #row-by-row generation

    #only accept anchors that appear in a significant number of docs
    print "identifying candidate anchors"
    candidate_anchors = []
    for i in xrange(V):
        if len(np.nonzero(row_M[i, :])[1]) > params.anchor_thresh:
            candidate_anchors.append(i)
    print len(candidate_anchors), "candidates"

    #_, anchors = gs.Projection_Find(Q, K, candidate_anchors)
    anchors = range(50)
    print "anchors are:", anchors
    anchor_file = file(outfile+'.anchors', 'w')
    print >>anchor_file, "\t".join(["topic id", "word id", "word"])
    for i, a in enumerate(anchors):
        print i, vocab[a]
        print >>anchor_file, "\t".join([str(x) for x in (i,a,vocab[a])])

    anchor_file.close()

    #recover topics
    row_sums = np.array(row_M.sum(1)).reshape(V)
    #generate Q_matrix rows for anchors
    Q_A = np.vstack(generate_Q_matrix(row_M, col_M, row_normalize=True, indices=anchors, projection_matrix=None))

    #build generator for rest of the Q_matrix rows
    Q = generate_Q_matrix(row_M, col_M, row_normalize=True, projection_matrix=None)
    A, topic_likelihoods = nonNegativeRecover(Q, Q_A, row_sums, anchors, params, loss)
    print "done recovering"

    np.savetxt(outfile+".A", A)
    np.savetxt(outfile+".topic_likelihoods", topic_likelihoods)

#display
    f = file(outfile+".topwords", 'w')
    for k in xrange(K):
        topwords = np.argsort(A[:, k])[-params.top_words:][::-1]
        print vocab[anchors[k]], ':',
        print >>f, vocab[anchors[k]], ':',
        for w in topwords:
            print vocab[w],
            print >>f, vocab[w],
        print ""
        print >>f, ""
