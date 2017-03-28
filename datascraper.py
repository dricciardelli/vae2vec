import re
from nltk.corpus import wordnet as wn
from nltk.corpus import words
from datamuse import datamuse
import numpy as np
import scipy.io as sio 
from wordnik import *


dct = {}

apiKey = '1494995aa73a3b9dea00502e50103839d32159b8e093cbcc1'
apiUrl = 'http://api.wordnik.com/v4'

client = swagger.ApiClient(apiKey, apiUrl)

wordApi = WordApi.WordApi(client)

training_set = []

for word in words.words():
	all_definitions = wordApi.getDefinitions(word)
	if all_definitions != None:
		for definition in all_definitions:
			training_set.append((word, definition.text))

dct['training_set'] = training_set

sio.savemat('definitions.mat', dct)

