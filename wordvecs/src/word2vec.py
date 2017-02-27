"""
Functions for preprocessing the data and loading the pre-trained word vectors.
"""

import sys
import numpy as np
from scipy.spatial.distance import cosine



__author__ = "jverma"


def compute_vecs_from_words(word2vec_file, vocab):
	"""
    Loads 300x1 word vecs from Google (Mikolov) word2vec.

    Parameters
    ----------
    word2vec_file: Path to the word2vec binary file.
    vocab: A list containing words in the vocab.

    Returns
    -------
    A dictionary containing words and their word embeddings (as numpy arrays).
    """
    word_vecs = {}
    with open(word2vec_file, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size

        print vocab_size, layer1_size, binary_len

        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs




##############################################################
##############################################################


def compute_words_from_vecs(word2vec_file, vec, n):
	"""
	Compute words closest to the input vector.

	Parameters
    ----------
    word2vec_file: Path to the word2vec binary file.
    vec: A 300-d input vector.
    n: Number of closest words to be retrieved.

    Returns
    -------
    A dictionary containing words and their distances from the input vector.
	"""
	word_dist_dict = {}

	with open(word2vec_file, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size

        print vocab_size, layer1_size, binary_len

        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
           word_vec = np.fromstring(f.read(binary_len), dtype='float32')
           dist = cosine(vec, word_vec)
           word_dist_dict[word] = dist
    # sort the dict by the distance values
    sorted_word_dist_dict = sorted(word_dist_dict.items(), key=operator.itemgetter(1))
    top_results = {}
    for u,v in sorted_word_dist_dict:
    	top_results[u] = v
    	if (len(top_results) > 10):
    		break
    return top_results












