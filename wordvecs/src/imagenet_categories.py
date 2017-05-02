"""
Understand the Imagenet categories using word reps.
"""
import sys
import os
import numpy as np 



all_words = set()
def get_category_dict(category_file):
    """
    """
    global all_words
    category_dict = {}
    for i,line in enumerate(category_file):
        record = line.rstrip().replace(',','').split()
        category_dict[record[0]] = record[1:]
        words = set(record[1:])
        all_words = all_words.union(words)
    return category_dict





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

            print(vocab_size, layer1_size, binary_len)

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



# word2vec_file = sys.argv[1]
category_file = open('synset_words.txt')
fname = '/Users/%s/Dropbox/irshad2janu/deeplearning_datasets/wordvectors/GoogleNews-vectors-negative300.bin'%os.getlogin()
categories = get_category_dict(category_file)
embedding_index = compute_vecs_from_words(fname, all_words)
print(len(embedding_index))

