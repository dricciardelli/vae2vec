import re
from nltk.corpus import wordnet as wn
from nltk.corpus import words
from datamuse import *
import numpy as np
import scipy.io as sio 

api = datamuse.Datamuse()


def grab_all_words(definition):
	lst = []
	all_words = api.words(ml=definition)
	for elem in all_words:
		lst.append((definition, elem['word']))
	return lst

def clean(s):
    # Cleans a string: Lowercasing, trimming, removing non-alphanumeric
    tmp = " ".join(re.findall(r'\w+', s, flags=re.UNICODE | re.LOCALE)).lower()
    return tmp.encode('ascii', 'ignore')
training_set = []

reverse_set = []

for word in words.words():
	clean_word = clean(word)
	all_definitions = wn.synsets(word)
	for definition in all_definitions:
		clean_definition = clean(definition.definition())
		reverse_set += grab_all_words(clean_definition)
		training_set.append((clean_word, clean_definition))
training_set = np.array(training_set)

dct = {}
dct['training'] = training_set
dct['reverse'] = reverse_set
sio.savemat('data.mat', dct)
