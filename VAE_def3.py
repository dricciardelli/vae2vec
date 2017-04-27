''' Significant lifting from https://jmetzen.github.io/2015-11-27/vae.html '''
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn
import random

import matplotlib.pyplot as plt

import re, string
from sklearn.feature_extraction.text import CountVectorizer

from collections import defaultdict
import pickle as pkl
import itertools

import ctc_loss

import os

def load_text(n,num_samples=None):
	# fname = 'Oxford_English_Dictionary.txt'
	# txt = []
	# with open(fname,'rb') as f:
	# 	txt = f.readlines()

	# txt = [x.decode('utf-8').strip() for x in txt]
	# txt = [re.sub(r'[^a-zA-Z ]+', '', x) for x in txt if len(x) > 1]

	# List of words
	# word_list = [x.split(' ', 1)[0].strip() for x in txt]
	# # List of definitions
	# def_list = [x.split(' ', 1)[1].strip()for x in txt]
	with open('./training_data/training_data.pkl','rb') as raw:
		word_list,dl=pkl.load(raw)
	def_list=[]	
	# def_list=[' '.join(defi) for defi in def_list]
	i=0
	while i<len( dl):
		defi=dl[i]
		if len(defi)>0:
			def_list+=[' '.join(defi)]
			i+=1
		else:
			dl.pop(i)
			word_list.pop(i)


	maxlen=0
	minlen=100
	for defi in def_list:
		minlen=min(minlen,len(defi.split()))
		maxlen=max(maxlen,len(defi.split()))
	print(minlen)
	print(maxlen)
	maxlen=30

	# # Initialize the "CountVectorizer" object, which is scikit-learn's
	# # bag of words tool.  
	# vectorizer = CountVectorizer(analyzer = "word",   \
	#                              tokenizer = None,    \
	#                              preprocessor = None, \
	#                              stop_words = None,   \
	#                              max_features = None, \
	#                              token_pattern='\\b\\w+\\b') # Keep single character words

	_map,rev_map=get_one_hot_map(word_list,def_list,n)

	if num_samples is not None:
		num_samples=len(word_list)
	# X = (36665, 56210)

	X = map_one_hot(word_list[:num_samples],_map,1,n)
	# y = (36665, 56210)
	# print _map
	y,mask = map_one_hot(def_list[:num_samples],_map,maxlen,n)
	print (np.max(y))
	return X, y, mask,rev_map

def get_one_hot_map(to_def,corpus,n):
	# words={}
	# for line in to_def:
	# 	if line:
	# 		words[line.split()[0]]=1

	

	# counts=defaultdict(int)
	# uniq=defaultdict(int)
	# for line in corpus:
	# 	for word in line.split():
	# 		if word not in words:
	# 			counts[word]+=1
	# words=list(words.keys())
	words=[]
	counts=defaultdict(int)
	uniq=defaultdict(int)
	for line in to_def+corpus:
		for word in line.split():
			if word not in words:
				counts[word]+=1
	_map=defaultdict(lambda :n+1)
	rev_map=defaultdict(lambda:"<UNK>")
	# words=words[:25000]
	for i in counts.values():
		uniq[i]+=1
	print (len(words))
	# random.shuffle(words)

	words+=list(map(lambda z:z[0],reversed(sorted(counts.items(),key=lambda x:x[1]))))[:n-len(words)]
	print (len(words))
	i=0
	# random.shuffle(words)
	for num_bits in range(binary_dim):
		for bit_config in itertools.combinations_with_replacement(range(binary_dim),num_bits+1):
			bitmap=np.zeros(binary_dim)
			bitmap[np.array(bit_config)]=1
			num=bitmap*(2** np.arange(binary_dim ))
			num=np.sum(num).astype(np.uint32)
			word=words[i]
			_map[word]=num
			rev_map[num]=word
			i+=1
			if i>=len(words):
				break
		if i>=len(words):
				break
	# 	for word in words:
	# 		i+=1
	# 		_map[word]=i
	# 		rev_map[i]=word
	rev_map[n+1]='<UNK>'
	if zero_end_tok:
		rev_map[0]='.'
	else:
		rev_map[0]='Start'
		rev_map[n+2]='End'
	print (list(reversed(sorted(uniq.items()))))
	print (len(list(uniq.items())))
	# print rev_map
	return _map,rev_map
def map_word_emb(corpus,_map):
	### NOTE: ONLY WORKS ON TARGET WORD (DOES NOT HANDLE UNK PROPERLY)
	rtn=[]
	rtn2=[]
	for word in corpus:
		mapped=_map[word]
		rtn.append(mapped)
		if get_rand_vec:
			mapped_rand=random.choice(list(_map.keys()))
			while mapped_rand==word:
				mapped_rand=random.choice(list(_map.keys()))
			mapped_rand=_map[mapped_rand]
			rtn2.append(mapped_rand)
	if get_rand_vec:
		return np.array(rtn),np.array(rtn2)
	return np.array(rtn)

def map_one_hot(corpus,_map,maxlen,n):
	if maxlen==1:
		if not form2:
			total_not=0
			rtn=np.zeros([len(corpus),n+3],dtype=np.float32)
			for l,line in enumerate(corpus):
				if len(line)==0:
					rtn[l,-1]=1
				else:
					mapped=_map[line]
					if mapped==75001:
						total_not+=1
					rtn[l,mapped]=1
			print (total_not,len(corpus))
			return rtn
		else:
			total_not=0
			if not onehot:
				rtn=np.zeros([len(corpus),binary_dim],dtype=np.float32)
			else:
				rtn=np.zeros([len(corpus),2**binary_dim],dtype=np.float32)
			for l,line in enumerate(corpus):
			# if len(line)==0:
			# 	rtn[l]=n+2
			# else:
			# 	if line not in _map:
			# 		total_not+=1
				mapped=_map[line]
				if mapped==75001:
					total_not+=1
				if onehot:
					binrep=np.zeros(2**binary_dim)
					print line
					binrep[mapped]=1
				else:
					binrep=(1&(mapped/(2**np.arange(binary_dim))).astype(np.uint32)).astype(np.float32)
				rtn[l]=binrep
			print (total_not,len(corpus))
			return rtn
	else:
		if form2:
			rtn=np.zeros([len(corpus),maxlen+2,binary_dim],dtype=np.float32)
		else:
			rtn=np.zeros([len(corpus),maxlen+2],dtype=np.int32)
		print (rtn.shape)
		mask=np.zeros([len(corpus),maxlen+2],dtype=np.float32)
		print (mask.shape)
		mask[:,1]=1.0
		totes=0
		nopes=0
		wtf=0
		for l,_line in enumerate(corpus):
			x=0
			line=_line.split()
			for i in range(min(len(line),maxlen)):
				# if line[i] not in _map:
				# 	nopes+=1

				mapped=_map[line[i]]
				if form2:
					binrep=(1&(mapped/(2**np.arange(binary_dim))).astype(np.uint32)).astype(np.float32)
					rtn[l,i+1,:]=binrep
				else:
					rtn[l,i+1]=mapped
				if mapped==75001:
					wtf+=1
				mask[l,i+1]=1.0
				totes+=1
				x=i+1
			to_app=n+2
			if zero_end_tok:
				to_app=0
			if form2:
				rtn[l,x+1,:]=(1&(to_app/(2**np.arange(binary_dim))).astype(np.uint32)).astype(np.float32)
			else:
				rtn[l,x+1]=to_app
			mask[l,x+1]=1.0

		print (nopes,totes,wtf)
		return rtn,mask


def xavier_init(fan_in, fan_out, constant=1e-4): 
	""" Xavier initialization of network weights"""
	# https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
	low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
	high = constant*np.sqrt(6.0/(fan_in + fan_out))
	return tf.random_uniform((fan_in, fan_out), 
							 minval=low, maxval=high, 
							 dtype=tf.float32)

class VariationalAutoencoder(object):
	""" Variation Autoencoder (VAE) with an sklearn-like interface implemented using TensorFlow.
	
	This implementation uses probabilistic encoders and decoders using Gaussian 
	distributions and  realized by multi-layer perceptrons. The VAE can be learned
	end-to-end.
	
	See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.
	"""
	def __init__(self, network_architecture, transfer_fct=tf.nn.softplus, 
				 learning_rate=0.001, batch_size=100,generative=False,ctrain=False,test=False,global_step=None):
		self.network_architecture = network_architecture
		self.transfer_fct = transfer_fct
		self.learning_rate = learning_rate
		print self.learning_rate
		self.batch_size = batch_size
		if global_step is None:
			global_step=tf.Variable(0,trainiable=False)
		self.global_step=global_step
		self.no_reload=[self.global_step]
		
		# tf Graph input
		self.n_words=network_architecture['n_input']
		if not form2:
			self.x = tf.placeholder(tf.float32, [None,self.n_words],name='x_in')
		else:
			self.x = tf.placeholder(tf.float32, [None,self.n_words],name='x_in')
		self.intype=type(self.x)
		if not form2:
			self.caption_placeholder = tf.placeholder(tf.int32, [None,network_architecture["maxlen"]],name='caption_placeholder')
		else:
			self.caption_placeholder = tf.placeholder(tf.float32, [None, network_architecture["maxlen"],self.n_words],name='caption_placeholder')
			print self.caption_placeholder.shape
		self.mask=tf.placeholder(tf.float32, [None, network_architecture["maxlen"]],name='mask')
		
		# Create autoencoder network
		to_restore=None
		if not generative:	
			self._create_network()
			# Define loss function based variational upper-bound and 
			# corresponding optimizer
			to_restore=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
			self.untrainable_variables=[x for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if x not in self.no_reload]
			for var in self.untrainable_variables:
				if var not in self.var_embs:
					var.trainable=embeddings_trainable
			self._create_loss_optimizer()

			self.test=test
				
		else:
			self._build_gen()
		

			

		# Initializing the tensor flow variables

		init = tf.global_variables_initializer()

		# Launch the session
		self.sess = tf.InteractiveSession()
		if embeddings_trainable:
			print [x.name for x in to_restore]
			self.saver = tf.train.Saver(var_list=to_restore,max_to_keep=100)
			saved_path=tf.train.latest_checkpoint(model_path)
		else:
			print [x.name for x in self.untrainable_variables]
			self.saver= tf.train.Saver(var_list=self.untrainable_variables,max_to_keep=100)
			saved_path=tf.train.latest_checkpoint(model_path.replace('vaedef','defdef'))
		self.sess.run(init)
		if ctrain:
			self.saver.restore(self.sess, saved_path)
		self.saver=tf.train.Saver(max_to_keep=100)
		print [x.name for x in self.saver._var_list]
	def _create_network(self):
		# Initialize autoencode network weights and biases
		network_weights = self._initialize_weights(**self.network_architecture)
		start_token_tensor=tf.constant((np.zeros([self.batch_size,binary_dim])).astype(np.float32),dtype=tf.float32)
		self.network_weights=network_weights
		seqlen=tf.cast(tf.reduce_sum(self.mask,reduction_indices=-1),tf.int32)
		

		

		KLD_penalty=tf.tanh(tf.cast(self.global_step,tf.float32)/1600.0)

		# Use recognition network to determine mean and 
		# (log) variance of Gaussian distribution in latent
		# space
		if not same_embedding:
			input_embedding,input_embedding_KLD_loss=self._get_input_embedding([network_weights['variational_encoding'],network_weights['biases_variational_encoding']],network_weights['input_meaning'])
		else:
			input_embedding,input_embedding_KLD_loss=self._get_input_embedding([network_weights['variational_encoding'],network_weights['biases_variational_encoding']],network_weights['LSTM'])


		state = self.lstm.zero_state(self.batch_size, dtype=tf.float32)

		loss = 0
		self.debug=0
		probs=[]
		with tf.variable_scope("RNN"):
			for i in range(self.network_architecture['maxlen']): 
				if i > 0:

					# current_embedding = tf.nn.embedding_lookup(self.word_embedding, caption_placeholder[:,i-1]) + self.embedding_bias
					if form2:
						current_embedding,KLD_loss = self._get_word_embedding([network_weights['variational_encoding'],network_weights['biases_variational_encoding']],network_weights['LSTM'], self.caption_placeholder[:,i-1,:],logit=True)
					else:
						current_embedding,KLD_loss = self._get_word_embedding([network_weights['variational_encoding'],network_weights['biases_variational_encoding']],network_weights['LSTM'], self.caption_placeholder[:,i-1])
					if transfertype2:
						current_embedding=tf.stop_gradient(current_embedding)
					loss+=tf.reduce_sum(KLD_loss*self.mask[:,i])*KLD_penalty
				else:
					 current_embedding = input_embedding
				if i > 0: 
					tf.get_variable_scope().reuse_variables()

				out, state = self.lstm(current_embedding, state)

				
				if i > 0: 
					if not form2:
						labels = tf.expand_dims(self.caption_placeholder[:, i], 1)
						ix_range=tf.range(0, self.batch_size, 1)
						ixs = tf.expand_dims(ix_range, 1)
						concat = tf.concat([ixs, labels],1)
						onehot = tf.sparse_to_dense(
								concat, tf.stack([self.batch_size, self.n_words]), 1.0, 0.0)
					else:
						onehot=self.caption_placeholder[:,i,:]

					logit = tf.matmul(out, network_weights['LSTM']['encoding_weight']) + network_weights['LSTM']['encoding_bias']
					if not use_ctc:
						if form2:
							# best_word=tf.nn.softmax(logit)
							
							# best_word=tf.round(best_word)
							# all_the_f_one_h.append(best_word)
							xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logit, labels=onehot)
							xentropy=tf.reduce_sum(xentropy,reduction_indices=-1)
						else:
							xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=onehot)

						
						xentropy = xentropy * self.mask[:,i]
						xentropy=tf.reduce_sum(xentropy)
						self.debug+=xentropy
						loss += xentropy

					else:
						probs.append(tf.expand_dims(tf.nn.sigmoid(logit),1))
			if not use_ctc:
				loss_ctc=0
			else:
				probs=tf.concat(probs,axis=1)
				probs=ctc_loss.get_output_probabilities(probs,self.caption_placeholder[:,1:,:])
				loss_ctc=ctc_loss.loss(probs,self.caption_placeholder[:,1:,:],self.network_architecture['maxlen']-2,self.batch_size,seqlen-1)
				self.debug=loss_ctc
			# self.debug/=tf.reduce_sum(self.mask[:,1:])
			loss = (loss / tf.reduce_sum(self.mask[:,1:]))+tf.reduce_sum(input_embedding_KLD_loss)/self.batch_size*KLD_penalty+loss_ctc

			self.loss=loss
	
	def _initialize_weights(self, n_lstm_input, maxlen, 
							n_input, n_z, n_z_m,n_z_m_2):
		all_weights = dict()
		if not same_embedding:
			all_weights['input_meaning'] = {
				'affine_weight': tf.Variable(xavier_init(n_z, n_lstm_input),name='affine_weight'),
				'affine_bias': tf.Variable(tf.zeros(n_lstm_input),name='affine_bias')}
		if not vanilla:
			all_weights['biases_variational_encoding'] = {
				'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32),name='out_meanb'),
				'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32),name='out_log_sigmab')}
			all_weights['variational_encoding'] = {
				'out_mean': tf.Variable(xavier_init(n_input, n_z),name='out_mean'),
				'out_log_sigma': tf.Variable(xavier_init(n_input, n_z),name='out_log_sigma')}
			
		else:
			all_weights['biases_variational_encoding'] = {
				'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32),name='out_meanb')}
			all_weights['variational_encoding'] = {
				'out_mean': tf.Variable(xavier_init(n_input, n_z),name='out_mean')}
			
		self.no_reload+=all_weights['input_meaning'].values()
		self.var_embs=[]
		if transfertype2:
			self.var_embs=all_weights['biases_variational_encoding'].values()+all_weights['variational_encoding'].values()

		self.lstm=tf.contrib.rnn.BasicLSTMCell(n_lstm_input)
		if lstm_stack>1:
			self.lstm=tf.contrib.rnn.MultiRNNCell([self.lstm]*lstm_stack)
		all_weights['LSTM'] = {
			'affine_weight': tf.Variable(xavier_init(n_z, n_lstm_input),name='affine_weight2'),
			'affine_bias': tf.Variable(tf.zeros(n_lstm_input),name='affine_bias2'),
			'encoding_weight': tf.Variable(xavier_init(n_lstm_input,n_input),name='encoding_weight'),
			'encoding_bias': tf.Variable(tf.zeros(n_input),name='encoding_bias'),
			'lstm': self.lstm}
		return all_weights
	
	def _get_input_embedding(self, ve_weights, aff_weights):
		z,vae_loss=self._vae_sample(ve_weights[0],ve_weights[1],self.x)
		embedding=tf.matmul(z,aff_weights['affine_weight'])+aff_weights['affine_bias']
		return embedding,vae_loss

	def _get_middle_embedding(self, ve_weights, lstm_weights, x,logit=False):
		if logit:
			z,vae_loss=self._vae_sample_mid(ve_weights[0],ve_weights[1],x)
		else:
			if not form2:
				z,vae_loss=self._vae_sample_mid(ve_weights[0],ve_weights[1],x, True)
			else:
				z,vae_loss=self._vae_sample(ve_weights[0],ve_weights[1],tf.one_hot(x,depth=self.network_architecture['n_input']))
				all_the_f_one_h.append(tf.one_hot(x,depth=self.network_architecture['n_input']))

		embedding=tf.matmul(z,lstm_weights['affine_weight'])+lstm_weights['affine_bias']
		return embedding,vae_loss

	def _get_word_embedding(self, ve_weights, lstm_weights, x,logit=False):
		if logit:
			z,vae_loss=self._vae_sample(ve_weights[0],ve_weights[1],x)
		else:
			if not form2:
				z,vae_loss=self._vae_sample(ve_weights[0],ve_weights[1],x, True)
			else:
				z,vae_loss=self._vae_sample(ve_weights[0],ve_weights[1],tf.one_hot(x,depth=self.network_architecture['n_input']))
				all_the_f_one_h.append(tf.one_hot(x,depth=self.network_architecture['n_input']))

		embedding=tf.matmul(z,lstm_weights['affine_weight'])+lstm_weights['affine_bias']
		return embedding,vae_loss
	

	def _vae_sample(self, weights, biases, x, lookup=False):
			#TODO: consider adding a linear transform layer+relu or softplus here first 
			if not lookup:
				mu=tf.matmul(x,weights['out_mean'])+biases['out_mean']
				if not vanilla:
					logvar=tf.matmul(x,weights['out_log_sigma'])+biases['out_log_sigma']
			else:
				mu=tf.nn.embedding_lookup(weights['out_mean'],x)+biases['out_mean']
				if not vanilla:
					logvar=tf.nn.embedding_lookup(weights['out_log_sigma'],x)+biases['out_log_sigma']

			if not vanilla:
				epsilon=tf.random_normal(tf.shape(logvar),name='epsilon')
				std=tf.exp(.5*logvar)
				z=mu+tf.multiply(std,epsilon)
			else:
				z=mu
			KLD=0.0
			if not vanilla:
				KLD = -0.5 * tf.reduce_sum(1 + logvar - tf.pow(mu, 2) - tf.exp(logvar),axis=-1)
				print logvar.shape,epsilon.shape,std.shape,z.shape,KLD.shape
			return z,KLD

	def _vae_sample_mid(self, weights, biases, x, lookup=False):
			#TODO: consider adding a linear transform layer+relu or softplus here first 
			if not lookup:
				mu=tf.matmul(x,weights['out_mean'])+biases['out_mean']
				if mid_vae:
					logvar=tf.matmul(x,weights['out_log_sigma'])+biases['out_log_sigma']
			else:
				mu=tf.nn.embedding_lookup(weights['out_mean'],x)+biases['out_mean']
				if mid_vae:
					logvar=tf.nn.embedding_lookup(weights['out_log_sigma'],x)+biases['out_log_sigma']

			if mid_vae:
				epsilon=tf.random_normal(tf.shape(logvar),name='epsilon')
				std=tf.exp(.5*logvar)
				z=mu+tf.multiply(std,epsilon)
			else:
				z=mu
			KLD=0.0
			if mid_vae:
				KLD = -0.5 * tf.reduce_sum(1 + logvar - tf.pow(mu, 2) - tf.exp(logvar),axis=-1)
				print logvar.shape,epsilon.shape,std.shape,z.shape,KLD.shape
			return z,KLD

	def _create_loss_optimizer(self):
		
		if clip_grad:
			opt_func = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)

			tvars = tf.trainable_variables()

			grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), .1)

			self.optimizer = opt_func.apply_gradients(zip(grads, tvars))
		else:
			self.optimizer = \
				tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

	def _create_loss_test(self):
		self.test_op = \
			tf.test.compute_gradient_error(self.x,np.array([self.batch_size,self.n_words]),self.loss,[1],extra_feed_dict={})
		
	def partial_fit(self, X,y,mask,testify=False):
		"""Train model based on mini-batch of input data.
		
		Return cost of mini-batch.
		"""
		if self.test and testify:
			print tf.test.compute_gradient_error(self.x,np.array([self.batch_size,self.n_words]),self.loss,[self.batch_size],extra_feed_dict={self.caption_placeholder: y, self.mask: mask})
			exit()
		else:
			opt, cost,shit = self.sess.run((self.optimizer, self.loss,self.debug), 
									  feed_dict={self.x: X, self.caption_placeholder: y, self.mask: mask})
			# print shit
		return cost,shit
		
	def _build_gen(self):
		#same setup as `_create_network` function 
		network_weights = self._initialize_weights(**self.network_architecture)
		if form2:
			start_token_tensor=tf.constant((np.zeros([self.batch_size,binary_dim])).astype(np.float32),dtype=tf.float32)
		else:
			start_token_tensor=tf.constant((np.zeros([self.batch_size])).astype(np.int32),dtype=tf.int32)
		self.network_weights=network_weights
		if not same_embedding:
			input_embedding,_=self._get_input_embedding([self.network_weights['variational_encoding'],self.network_weights['biases_variational_encoding']],self.network_weights['input_meaning'])
		else:
			input_embedding,_=self._get_input_embedding([self.network_weights['variational_encoding'],self.network_weights['biases_variational_encoding']],self.network_weights['LSTM'])
		print input_embedding.shape
		# image_embedding = tf.matmul(img, self.img_embedding) + self.img_embedding_bias
		state = self.lstm.zero_state(self.batch_size,dtype=tf.float32)

		#declare list to hold the words of our generated captions
		all_words = []
		with tf.variable_scope("RNN"):
			# in the first iteration we have no previous word, so we directly pass in the image embedding
			# and set the `previous_word` to the embedding of the start token ([0]) for the future iterations
			output, state = self.lstm(input_embedding, state)
			print state,output.shape
			if form2:
				previous_word,_ = self._get_word_embedding([self.network_weights['variational_encoding'],self.network_weights['biases_variational_encoding']],self.network_weights['LSTM'], start_token_tensor,logit=True)
			else:
				previous_word,_ = self._get_word_embedding([self.network_weights['variational_encoding'],self.network_weights['biases_variational_encoding']],self.network_weights['LSTM'], start_token_tensor)
			print previous_word.shape
			# previous_word = tf.nn.embedding_lookup(self.word_embedding, [0]) + self.embedding_bias

			for i in range(self.network_architecture['maxlen']):
				tf.get_variable_scope().reuse_variables()
				print i

				out, state = self.lstm(previous_word, state)


				# get a one-hot word encoding from the output of the LSTM
				logit=tf.matmul(out, network_weights['LSTM']['encoding_weight']) + network_weights['LSTM']['encoding_bias']
				if not form2:
					best_word = tf.argmax(logit, 1)
				else:
					best_word=tf.nn.sigmoid(logit)
					best_word=tf.round(best_word)

				# with tf.device("/cpu:0"):
				#     # get the embedding of the best_word to use as input to the next iteration of our LSTM 
				#     previous_word = tf.nn.embedding_lookup(self.word_embedding, best_word)

				# previous_word += self.embedding_bias
				print logit.shape
				if form2:
					previous_word,_ = self._get_word_embedding([self.network_weights['variational_encoding'],self.network_weights['biases_variational_encoding']],self.network_weights['LSTM'], best_word,logit=True)
				else:
					previous_word,_ = self._get_word_embedding([self.network_weights['variational_encoding'],self.network_weights['biases_variational_encoding']],self.network_weights['LSTM'], best_word)
				print previous_word.shape
				all_words.append(best_word)

		self.generated_words=all_words

	def generate(self, _map, x):
		""" Generate data by sampling from latent space.
		
		If z_mu is not None, data for this point in latent space is
		generated. Otherwise, z_mu is drawn from prior in latent 
		space.        
		# """
		# if z_mu is None:
		#     z_mu = np.random.normal(size=self.network_architecture["n_z"])
		# # Note: This maps to mean of distribution, we could alternatively
		# # sample from Gaussian distribution
		# return self.sess.run(self.x_reconstr_mean, 
		#                      feed_dict={self.z: z_mu})
		
		# saver = tf.train.Saver()
		# saver.restore(self.sess, tf.train.latest_checkpoint(model_path))

		generated_word_index,f_it= self.sess.run([self.generated_words,all_the_f_one_h], feed_dict={self.x:x})
		print f_it
		print generated_word_index
		if form2:
			generated_word_index=np.array(bin_to_int(generated_word_index))
			generated_word_index=np.rollaxis(generated_word_index,1)
		else:
			generated_word_index=np.array(generated_word_index)


		return generated_word_index

		# generated_sentence = ixtoword(_map,generated_word_index)
		# return generated_sentence

def ixtoword(_map,ixs):
	return [[_map[x] for x in y] for y in ixs]
def bin_to_int(a):
	return [(x*(2** np.arange(x.shape[-1] ))).sum(axis=-1).astype(np.uint32) for x in a]
	

def train(network_architecture, learning_rate=0.001,
		  batch_size=100, training_epochs=10, display_step=2,gen=False,ctrain=False,test=False):
	if should_decay and not gen:
		global_step=tf.Variable(0,trainable=False)
		learning_rate = tf.train.exponential_decay(learning_rate, global_step,
                                           all_samps, 0.95, staircase=True)
	vae = VariationalAutoencoder(network_architecture, 
								 learning_rate=learning_rate, 
								 batch_size=batch_size,generative=gen,ctrain=ctrain,test=test,global_step=global_step)
	# Training cycle
	# if test:
	# 	maxlen=network_architecture['maxlen']
	# 	return tf.test.compute_gradient_error([vae.x,vae.caption_placeholder,vae.mask],[np.array([batch_size,n_input]),np.array([batch_size,maxlen,n_input]),np.array([batch_size,maxlen])],vae.loss,[])
	if gen:
		return vae
	costs=[]
	indlist=np.arange(all_samps).astype(int)
	for epoch in range(training_epochs):
		avg_cost = 0.
		total_batch = int(n_samples / batch_size)
		# Loop over all batches
		
		np.random.shuffle(indlist)
		testify=False
		avg_loss=0
		for i in range(total_batch):

			batch_xs = X[indlist[i*batch_size:(i+1)*batch_size]]

			# Fit training using batch data
			if epoch==2 and i ==0:
				testify=True
			cost,loss = vae.partial_fit(batch_xs,y[indlist[i*batch_size:(i+1)*batch_size]].astype(np.uint32),mask[indlist[i*batch_size:(i+1)*batch_size]],testify=testify)

			# Compute average loss
			avg_cost = avg_cost * i /(i+1) +cost/(i+1)
			avg_loss=avg_loss*i/(i+1)+cost/(i+1)
			# if i% display_step==0:
			print avg_cost,avg_loss
			if epoch==0 and i==0:
				costs.append(avg_cost)

		
		costs.append(avg_cost)
		
		# Display logs per epoch step
		if epoch % display_step == 0 or epoch==1:
			if should_save:
				print 'saving'
				vae.saver.save(vae.sess, os.path.join(model_path,'model'))
				pkl.dump(costs,open(loss_output_path,'wb'))
			print("Epoch:", '%04d' % (epoch+1), 
				  "cost=", avg_cost)


	return vae

if __name__ == "__main__":
	import sys
	form2=True
	vanilla=True
	if sys.argv[1]!='vanilla':
		vanilla=False
	mid_vae=False
	if sys.argv[2]=='mid_vae':
		mid_vae=True
	same_embedding=False
	clip_grad=True
	if sys.argv[3]!='clip':
		clip_grad=False
	should_save=True
	should_train=True
	# should_train=not should_train
	should_continue=False
	should_decay=True
	zero_end_tok=True
	training_epochs=int(sys.argv[14])
	batch_size=int(sys.argv[4])
	onehot=False
	embeddings_trainable=False
	if sys.argv[5]!='transfer':
		print('not transfering')
		embeddings_trainable=True
	transfertype2=True
	binary_dim=int(sys.argv[6])
	all_the_f_one_h=[]
	if not zero_end_tok:
		X, y, mask, _map = load_text(2**binary_dim-4)
	else:
		X, y, mask, _map = load_text(2**binary_dim-3)
	n_input =binary_dim
	n_samples = 30000
	lstm_dim=int(sys.argv[7])
	model_path = sys.argv[8]
	vartype=''
	transfertype=''
	maxlen=int(sys.argv[9])+2
	n_z=int(sys.argv[10])
	n_z_m=int(sys.argv[11])
	n_z_m_2=int(sys.argv[12])
	if sys.argv[13]!='2':
		transfertype2=False
	if not vanilla:
		vartype='var'
	if not embeddings_trainable:
		transfertype='transfer'
		if transfertype2:
			transfertype+='2'
	cliptype=''
	if clip_grad:
		cliptype='clip'
	use_ctc=False
	losstype=''
	if sys.argv[15]=='ctc_loss':
		use_ctc=True
		losstype='ctc'
	lstm_stack=int(sys.argv[16])
	use_bdlstm=False

	loss_output_path= 'losses/%ss_%sb_%sl_%sh_%sd_%sz_%szm_%s%s%svaedef%s.pkl'%(str(lstm_stack),str(batch_size),str(maxlen-2),str(lstm_dim),str(n_input),str(n_z),str(n_z_m),str(losstype),str(cliptype),str(vartype),str(transfertype))
	all_samps=len(X)
	n_samples=all_samps
	# X, y = X[:n_samples, :], y[:n_samples, :]

	network_architecture = \
		dict(maxlen=maxlen, # 2nd layer decoder neurons
			 n_input=n_input, # One hot encoding input
			 n_lstm_input=lstm_dim, # LSTM cell size
			 n_z=n_z, # dimensionality of latent space
			 n_z_m=n_z_m,
			 n_z_m_2=n_z_m_2
			 ) 


	if should_train:
		# vae_2d = train(network_architecture, training_epochs=training_epochs, batch_size=batch_size,gen=False,ctrain=should_continue)
		# print train(network_architecture, training_epochs=training_epochs, batch_size=batch_size,gen=False,ctrain=should_continue,test=True)
		vae_2d = train(network_architecture, training_epochs=training_epochs, batch_size=batch_size,gen=False,ctrain=should_continue,learning_rate=.005)
	else:
		vae_2d = train(network_architecture, training_epochs=training_epochs, batch_size=batch_size,gen=True,ctrain=True)
	
	# #	vae_2d._build_gen()
		ind_list=np.arange(len(X)).astype(int)
		np.random.shuffle(ind_list)
		x_sample = X[ind_list[:1000]]
		print x_sample
		y_sample = y[ind_list[:1000]]
		print y_sample

		y_hat = vae_2d.generate(_map,x_sample)
		y_hat=y_hat[:10]
		# print y_hat
		y_hat_words=ixtoword(_map,y_hat)
		print y_hat_words
		if form2:
			y_words=ixtoword(_map,np.array(bin_to_int(y_sample[:10])))
		else:
			y_words=ixtoword(_map,y_sample)

		print(y_hat)
		print(y_hat_words)
		print(y_words)
	# 	# plt.figure(figsize=(8, 6)) 
		# plt.scatter(z_mu[:, 0], z_mu[:, 1], c=np.argmax(y_sample, 1))
		# plt.colorbar()
		# plt.grid()
		# plt.show()
