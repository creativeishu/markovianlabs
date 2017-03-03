"""
Functions for preprocessing the data and loading the pre-trained word vectors.
"""

import sys
import numpy as np
from scipy.spatial.distance import cosine
import cPickle



__author__ = "jverma"


def build_word_vec_dict(word2vec_file):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec.

    Parameters
    ----------
    word2vec_file: Path to the word2vec binary file.
    
    Returns
    -------
    A pickle-serialized dictionary containing words and their word embeddings (as numpy arrays).
    """    
    word_vecs = {}
    with open(word2vec_file, "rb") as f:
            header = f.readline()
            vocab_size, layer1_size = map(int, header.split())
            binary_len = np.dtype('float32').itemsize * layer1_size

            print(vocab_size)
            print(layer1_size)
            print(binary_len)

            for line in xrange(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == ' ':
                        word = ''.join(word)
                        break
                    if ch != '\n':
                        word.append(ch)   
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32') 
    pickle_file_name = "../data/wordvec_dict.p"
    cPickle.dump([word_vecs], open(pickle_file_name, "wb"))
    print("pickle created....")
    return None





def compute_word_vec(pickle_file, word):
    """
    Extracts word embeddings for a given word.
    """
    wordvec_dict = cPickle.load(open(pickle_file))
    return wordvec_dict[word]




def compute_vecs_from_words(word2vec_file, vocab):
    """
    Extracts word embeddings for words in a given list.

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


word2vec_file = sys.argv[1]
build_word_vec_dict(word2vec_file)
# vocab = ['apple', 'banana']
# xx = compute_vecs_from_words(word2vec_file, vocab)

# import matplotlib.pyplot as plt
# plt.plot(xx['apple'], xx['banana'])

# plt.show()








