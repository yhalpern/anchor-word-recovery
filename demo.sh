#!/bin/bash
echo "start demo"
loss="L2"
K="50"
corpus="nips"
rareWordThresh="50"

echo "downloading UCI $corpus corpus"
#rm vocab.$corpus.txt
#wget http://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/vocab.$corpus.txt
#wget http://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/docword.$corpus.txt.gz
#gunzip docword.$corpus.txt.gz

echo "preprocessing, translate from docword.txt to scipy format"
#python uci_to_scipy.py docword.$corpus.txt M_$corpus.full_docs.mat

echo "preprocessing: removing rare words and stopwords"
echo "removing words that do not appear in more than $rareWordThresh documents"
#python truncate_vocabulary.py M_$corpus.full_docs.mat vocab.$corpus.txt $rareWordThresh

echo "learning with nonnegative recover method using $loss loss..."

python learn_topics.py M_$corpus.full_docs.mat.trunc.mat settings.example vocab.$corpus.txt.trunc $K $loss demo_$loss\_out.$corpus.$K
