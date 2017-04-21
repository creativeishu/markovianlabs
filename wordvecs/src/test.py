import sys

from text_processing_util import TextProcessing
from text_cnn import kimCNN


MAX_SEQUENCE_LENGTH = 500
MAX_NB_WORDS = 23000
EMBEDDING_DIM = 300
# VALIDATION_SPLIT = 0.0




train_file = sys.argv[1]
fname = sys.argv[2]


# Prepare text samples and their labels
print('Processing text dataset')
# labels_index = {'GOAL': 0, 'DEMO': 1, 'PREG': 2, 'SOCL': 3, 'FAML': 4, 'DISE': 5, 'TRMT': 6}
labels = []
texts = []
# titles = []
for line in open(train_file):
    record = line.rstrip().split('\t')
    # label_id = labels_index[record[0]]
    # labels.append(label_id)
    labels.append(record[0])
    # titles.append(record[1].rstrip().lower())
    texts.append(record[2].rstrip().lower())


print("Found %s texts" %len(texts))
print("Found %s labels" %len(labels))
# print('Found %s titles.' % len(titles))



tp = TextProcessing(texts, labels, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, MAX_NB_WORDS)

x_train, y_train, x_val, y_val, word_index = tp.preprocess()
embeddings_index = tp.build_embedding_index_from_word2vec(fname, word_index)
print('Found %s word vectors.' % len(embeddings_index))

labels_index = tp.labels_index

model = kimCNN(EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, MAX_NB_WORDS, embeddings_index, word_index, labels_index=labels_index)
print(model.summary())

model.fit(x=x_train, y=y_train, batch_size=50, epochs=25)
# , validation_data=(x_val, y_val))









