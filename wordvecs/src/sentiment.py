from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import cPickle
import numpy as np
import os


model_file = '/Users/%s/Dropbox/irshad2janu/deeplearning_datasets/wordvectors/sentiment/imdb_sentiment.h5'%os.getlogin()
model = load_model(model_file)
print(model.summary())

word_indices = '/Users/%s/Dropbox/irshad2janu/deeplearning_datasets/wordvectors/sentiment/tokenization_and_embedding'%os.getlogin()

test_texts = ['This is such a great movie.']

seqs = []
word_index, emdedding_index = cPickle.load(open(word_indices, 'rb'))
for text in test_texts:
	text = text.split(' ')
	text_seq = []
	for word in text:
		try:
			text_seq.append(word_index[word])
		except:
			pass
	# text_seq = pad_sequences(text_seq, maxlen=50)
	seqs.append(text_seq)

seqs = np.array(seqs)
seqs = pad_sequences(seqs, maxlen=50)
print(model.predict(seqs))
