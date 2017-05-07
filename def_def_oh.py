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
n=150000-2
def map_lambda():
	return n+1
def rev_map_lambda():
	return "<UNK>"
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
	# words={}
	while i<len( dl):
		defi=dl[i]
		if len(defi)>0:
			def_list+=[' '.join(defi)]
			i+=1
		else:
			dl.pop(i)
			word_list.pop(i)

	# for w,d in zip(word_list,def_list):
	# 	if w not in words:
	# 		words[w]=[]
	# 	words[w].append(d)
	# word_list=[]
	# def_list=[]
	# for word in words:
	# 	word_list.append(word)
	# 	# def_list.append(random.choice(words[word]))
	# 	def_list.append(words[word][0])

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
	pkl.dump(_map,open('mapaoh.pkl','wb'))
	pkl.dump(rev_map,open('rev_mapaoh.pkl','wb'))
	# exit()
	if num_samples is not None:
		num_samples=len(word_list)
	# X = (36665, 56210)

	X = map_one_hot(word_list[:num_samples],_map,1,n)
	# # y = (36665, 56210)
	# # print _map
	y,mask = map_one_hot(def_list[:num_samples],_map,maxlen,n)
	np.save('Xaoh',X)
	np.save('yaoh',y)
	np.save('maskaoh',mask)
	# X=np.load('Xa.npy','r')
	# y=np.load('ya.npy','r')
	# mask=np.load('maska.npy','r')
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
	_map=defaultdict(map_lambda)
	rev_map=defaultdict(rev_map_lambda)
	# words=words[:25000]
	for i in counts.values():
		uniq[i]+=1
	print (len(words))
	# random.shuffle(words)

	words+=list(map(lambda z:z[0],reversed(sorted(counts.items(),key=lambda x:x[1]))))[:n-len(words)]
	print (len(words))
	i=0
	# random.shuffle(words)
	# for num_bits in range(binary_dim):
	# 	for bit_config in itertools.combinations_with_replacement(range(binary_dim),num_bits+1):
	# 		bitmap=np.zeros(binary_dim)
	# 		bitmap[np.array(bit_config)]=1
	# 		num=bitmap*(2** np.arange(binary_dim ))
	# 		num=np.sum(num)
	# 		num=int(num)
	# 		word=words[i]
	# 		_map[word]=num
	# 		rev_map[num]=word
	# 		i+=1
	# 		if i>=len(words):
	# 			break
	# 	if i>=len(words):
	# 			break
	# i+=1
	for word in words:
		i+=1
		_map[word]=i
		rev_map[i]=word
	rev_map[n+1]='<UNK>'
	if zero_end_tok:
		rev_map[0]='.'
	else:
		rev_map[0]='Start'
		rev_map[2]='End'
	print (list(reversed(sorted(uniq.items()))))
	print (len(list(uniq.items())))
	print (len(rev_map.keys()))
	print(len(_map.keys()))
	print ('heyo')
	# print rev_map
	return _map,rev_map
def map_word_emb(corpus,_map):
	### NOTE: ONLY WORKS ON TARGET WORD (DOES NOT HANDLE UNK PROPERLY)
	rtn=[]
	rtn2=[]
	num_failed=0
	num_counted=0
	for word in corpus:
		w=word.lower()
		num_counted+=1
		if w not in _map:
			num_failed+=1
		mapped=_map[w]
		rtn.append(mapped)
		if get_rand_vec:
			mapped_rand=random.choice(list(_map.keys()))
			while mapped_rand==word:
				mapped_rand=random.choice(list(_map.keys()))
			mapped_rand=_map[mapped_rand]
			rtn2.append(mapped_rand)
	print 'fuck',num_failed/float(num_counted)
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
			rtn=np.zeros([len(corpus)],dtype=np.float32)
			for l,line in enumerate(corpus):
				if len(line)==0:
					rtn[l,-1]=1
				else:
					mapped=_map[line]
					if mapped==75001:
						total_not+=1
					rtn[l]=mapped
			print (total_not,len(corpus))
			return rtn
	else:
		if form2:
			rtn=np.zeros([len(corpus),maxlen+2],dtype=np.float32)
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
			for i in range(min(len(line),maxlen-1)):
				# if line[i] not in _map:
				# 	nopes+=1

				mapped=_map[line[i]]
				rtn[l,i+1]=mapped
				if mapped==n+1:
					wtf+=1
				mask[l,i+1]=1.0
				totes+=1
				x=i+1
			to_app=n+2
			if zero_end_tok:
				to_app=0
			
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
		self.timestep=tf.placeholder(tf.float32,[],name='timestep')
		# Create autoencoder network
		to_restore=None
		with tf.device('/cpu:0'):
			self.embw=tf.Variable(xavier_init(network_architecture['n_input'],network_architecture['n_z']),name='embw')
		self.embb=tf.Variable(tf.zeros([network_architecture['n_z']]),name='embb')
		if not generative:	
			self._create_network()
			# Define loss function based variational upper-bound and 
			# corresponding optimizer
			to_restore=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

			self._create_loss_optimizer()

			self.test=test
				
		else:
			self._build_gen()
		

			

		# Initializing the tensor flow variables

		init = tf.global_variables_initializer()

		# Launch the session
		self.sess = tf.InteractiveSession()
		if embeddings_trainable:
			self.saver = tf.train.Saver(var_list=to_restore,max_to_keep=100)
			saved_path=tf.train.latest_checkpoint(model_path)
		else:
			self.saver= tf.train.Saver(var_list=self.untrainable_variables,max_to_keep=100)
			mod_path=model_path
			if use_ctc:
				mod_path=mod_path[:-3]
			saved_path=tf.train.latest_checkpoint(mod_path.replace('defdef','embtransfer'))
		self.sess.run(init)
		if ctrain:
			self.saver.restore(self.sess, saved_path)
		self.saver=tf.train.Saver(max_to_keep=100)
	
	def _create_network(self):
		# Initialize autoencode network weights and biases
		network_weights = self._initialize_weights(**self.network_architecture)
		start_token_tensor=tf.constant((np.zeros([self.batch_size,binary_dim])).astype(np.float32),dtype=tf.float32)
		self.network_weights=network_weights
		seqlen=tf.cast(tf.reduce_sum(self.mask,reduction_indices=-1),tf.int32)
		self.embedded_input_KLD_loss=tf.constant(0.0)
		self.input_embedding_KLD_loss=tf.constant(0.0)
		# def train_encoder():
		embedded_input,self.embedded_input_KLD_loss=self._get_word_embedding([network_weights['variational_encoding'],network_weights['biases_variational_encoding']],network_weights['input_meaning'],tf.reshape(self.caption_placeholder,[-1,self.network_architecture['n_input']]),logit=True)
		embedded_input=tf.reshape(embedded_input,[-1,self.network_architecture['maxlen'],self.network_architecture['n_lstm_input']])
		if not vanilla:
			self.embedded_input_KLD_loss=tf.reshape(embedded_input_KLD_loss,[-1,self.network_architecture['maxlen']])[:,1:]
		encoder_input=embedded_input[:,1:,:]
		cell=tf.contrib.rnn.BasicLSTMCell(self.network_architecture['n_lstm_input'])
		if lstm_stack>1:
			cell=tf.contrib.rnn.MultiRNNCell([cell]*lstm_stack)
		if not use_bdlstm:
			encoder_outs,encoder_states=rnn.dynamic_rnn(cell,encoder_input,sequence_length=seqlen-1,dtype=tf.float32,time_major=False)
		else:
			backward_cell=tf.contrib.rnn.BasicLSTMCell(self.network_architecture['n_lstm_input'])
			if lstm_stack>1:
				backward_cell=tf.contrib.rnn.MultiRNNCell([backward_cell]*lstm_stack)
			encoder_outs,encoder_states=rnn.bidirectional_dynamic_rnn(cell,backward_cell,encoder_input,sequence_length=seqlen-1,dtype=tf.float32,time_major=False)
		ix_range=tf.range(0,self.batch_size,1)
		ixs=tf.expand_dims(ix_range,-1)
		to_cat=tf.expand_dims(seqlen-2,-1)
		gather_inds=tf.concat([ixs,to_cat],axis=-1)
		print encoder_outs
		outs=tf.gather_nd(encoder_outs,gather_inds)
		outs=tf.nn.dropout(outs,.75)
		self.deb=tf.gather_nd(self.caption_placeholder[:,1:,:],gather_inds)
		print outs.shape
		input_embedding,self.input_embedding_KLD_loss=self._get_middle_embedding([network_weights['middle_encoding'],network_weights['biases_middle_encoding']],network_weights['middle_encoding'],outs,logit=True)
			# return input_embedding
		# input_embedding=tf.nn.l2_normalize(input_embedding,dim=-1)
		self.other_loss=tf.constant(0,dtype=tf.float32)
		KLD_penalty=tf.tanh(tf.cast(self.timestep,tf.float32)/1.0)
		cos_penalty=tf.maximum(-0.1,tf.tanh(tf.cast(self.timestep,tf.float32)/(5.0)))

		self.input_KLD_loss=tf.constant(0.0)
		# def train_decoder():
		if form3:
			_x,self.input_KLD_loss=self._get_input_embedding([network_weights['variational_encoding'],network_weights['biases_variational_encoding']],network_weights['variational_encoding'])
			self.input_KLD_loss=tf.reduce_mean(self.input_KLD_loss)*KLD_penalty#*tf.constant(0.0,dtype=tf.float32)
			normed_embedding= tf.nn.l2_normalize(self.mid_var, dim=-1)
			normed_target=tf.nn.l2_normalize(self.word_var,dim=-1)
			cos_sim=(tf.reduce_sum(tf.multiply(normed_embedding,normed_target),axis=-1))
			# # self.exp_loss=tf.reduce_mean((-cos_sim))
			# # self.exp_loss=tf.reduce_sum(xentropy)/float(self.batch_size)
			self.other_loss += tf.reduce_mean(-(cos_sim))*cos_penalty
			# other_loss+=tf.reduce_mean(tf.reduce_sum(tf.square(_x-input_embedding),axis=-1))*cos_penalty
		return _x
		# input_embedding=tf.cond(tf.equal(self.timestep%5,0),train_decoder,train_encoder)
		# Use recognition network to determine mean and 
		# (log) variance of Gaussian distribution in latent
		# space
		# if not same_embedding:
		# 	input_embedding,input_embedding_KLD_loss=self._get_input_embedding([network_weights['variational_encoding'],network_weights['biases_variational_encoding']],network_weights['input_meaning'])
		# else:
		# 	input_embedding,input_embedding_KLD_loss=self._get_input_embedding([network_weights['variational_encoding'],network_weights['biases_variational_encoding']],network_weights['LSTM'])
		if not embeddings_trainable:
			input_embedding=tf.stop_gradient(input_embedding)
		# embed2decoder=tf.Variable(xavier_init(self.network_architecture['n_z_m_2'],self.network_architecture['n_lstm_input']),name='decoder_embedding_weight')
		# embed2decoder_bias=tf.Variable(tf.zeros(self.network_architecture['n_lstm_input']),name='decoder_embedding_bias')
		state = self.lstm.zero_state(self.batch_size, dtype=tf.float32)
		# input_embedding=tf.matmul(input_embedding,embed2decoder)+embed2decoder_bias
		loss = 0
		self.debug=0
		probs=[]
		with tf.variable_scope("RNN"):
			for i in range(self.network_architecture['maxlen']): 
				if i > 0:

					# current_embedding = tf.nn.embedding_lookup(self.word_embedding, caption_placeholder[:,i-1]) + self.embedding_bias
					if form4:
						current_embedding,KLD_loss=input_embedding,0
					elif form2:
						current_embedding,KLD_loss = self._get_word_embedding([network_weights['variational_encoding'],network_weights['biases_variational_encoding']],network_weights['LSTM'], self.caption_placeholder[:,i-1,:],logit=True)
					else:
						current_embedding,KLD_loss = self._get_word_embedding([network_weights['variational_encoding'],network_weights['biases_variational_encoding']],network_weights['LSTM'], self.caption_placeholder[:,i-1])
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
						onehot=self.caption_placeholder[:,i]

					logit = tf.matmul(out, network_weights['LSTM']['encoding_weight']) + network_weights['LSTM']['encoding_bias']
					if not use_ctc:
						
						xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=onehot)

						
						xentropy = xentropy * self.mask[:,i]
						xentropy=tf.reduce_sum(xentropy)
						self.debug+=xentropy
						loss += xentropy

					else:
						probs.append(tf.expand_dims(tf.nn.sigmoid(logit),1))
			self.debug=[input_KLD_loss,tf.reduce_mean(input_embedding_KLD_loss)/self.batch_size*KLD_penalty,other_loss,KLD_penalty]
			if not use_ctc:
				loss_ctc=0
				# self.debug=other_loss
				# self.debug=[input_KLD_loss,embedded_input_KLD_loss,input_embedding_KLD_loss]
			else:
				probs=tf.concat(probs,axis=1)
				probs=ctc_loss.get_output_probabilities(probs,self.caption_placeholder[:,1:,:])
				loss_ctc=ctc_loss.loss(probs,self.caption_placeholder[:,1:,:],self.network_architecture['maxlen']-2,self.batch_size,seqlen-1)
				self.debug=loss_ctc
			# 
			loss = (loss / tf.reduce_sum(self.mask[:,1:]))+tf.reduce_sum(self.input_embedding_KLD_loss)/self.batch_size*KLD_penalty+tf.reduce_sum(self.embedded_input_KLD_loss*self.mask[:,1:])/tf.reduce_sum(self.mask[:,1:])*KLD_penalty+loss_ctc+self.input_KLD_loss+self.other_loss

			self.loss=loss
	
	def _initialize_weights(self, n_lstm_input, maxlen, 
							n_input, n_z, n_z_m,n_z_m_2):
		all_weights = dict()
		if form3:
			n_in=n_z
		else:
			n_in=n_input
		if not same_embedding:
			all_weights['input_meaning'] = {
				'affine_weight': tf.Variable(xavier_init(n_z, n_lstm_input),name='affine_weight',trainable=embeddings_trainable),
				'affine_bias': tf.Variable(tf.zeros(n_lstm_input),name='affine_bias',trainable=embeddings_trainable)}
		# if not vanilla:
		all_weights['biases_variational_encoding'] = {
			'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32),name='out_meanb',trainable=embeddings_trainable),
			'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32),name='out_log_sigmab',trainable=embeddings_trainable)}
		all_weights['variational_encoding'] = {
			'out_mean': tf.Variable(xavier_init(n_in, n_z),name='out_mean',trainable=embeddings_trainable),
			'out_log_sigma': tf.Variable(xavier_init(n_in, n_z),name='out_log_sigma',trainable=embeddings_trainable),
			'affine_weight': tf.Variable(xavier_init(n_z, n_lstm_input),name='in_affine_weight'),
			'affine_bias': tf.Variable(tf.zeros(n_lstm_input),name='in_affine_bias')
			}
		# else:
		# 	all_weights['biases_variational_encoding'] = {
		# 		'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32),name='out_meanb',trainable=embeddings_trainable)}
		# 	all_weights['variational_encoding'] = {
		# 		'out_mean': tf.Variable(xavier_init(n_in, n_z),name='out_mean',trainable=embeddings_trainable),
		# 		'affine_weight': tf.Variable(xavier_init(n_z, n_lstm_input),name='in_affine_weight'),
		# 		'affine_bias': tf.Variable(tf.zeros(n_lstm_input),name='in_affine_bias')}
			
			
		self.untrainable_variables=all_weights['input_meaning'].values()+all_weights['biases_variational_encoding'].values()+all_weights['variational_encoding'].values()
		if mid_vae:
			all_weights['biases_middle_encoding'] = {
				'out_mean': tf.Variable(tf.zeros([n_z_m], dtype=tf.float32),name='mid_out_meanb'),
				'out_log_sigma': tf.Variable(tf.zeros([n_z_m], dtype=tf.float32),name='mid_out_log_sigmab')}
			all_weights['middle_encoding'] = {
				'out_mean': tf.Variable(xavier_init(n_lstm_input, n_z_m),name='mid_out_mean'),
				'out_log_sigma': tf.Variable(xavier_init(n_lstm_input, n_z_m),name='mid_out_log_sigma'),
				'affine_weight': tf.Variable(xavier_init(n_z_m, n_lstm_input),name='mid_affine_weight'),
				'affine_bias': tf.Variable(tf.zeros(n_lstm_input),name='mid_affine_bias')}
			all_weights['embmap']={
				'out_mean': tf.Variable(xavier_init(n_in, n_z),name='embmap_out_mean'),
				'out_log_sigma': tf.Variable(xavier_init(n_in, n_z),name='embmap_out_log_sigma')
			}
			all_weights['embmap_biases']={
				'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32),name='embmap_out_meanb',trainable=embeddings_trainable),
				'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32),name='embmap_out_log_sigmab',trainable=embeddings_trainable)
			}
		else:
			all_weights['biases_middle_encoding'] = {
				'out_mean': tf.Variable(tf.zeros([n_z_m], dtype=tf.float32),name='mid_out_meanb')}
			all_weights['middle_encoding'] = {
				'out_mean': tf.Variable(xavier_init(n_lstm_input, n_z_m),name='mid_out_mean'),
				'affine_weight': tf.Variable(xavier_init(n_z_m, n_lstm_input),name='mid_affine_weight'),
				'affine_bias': tf.Variable(tf.zeros(n_lstm_input),name='mid_affine_bias')}
			all_weights['embmap']={
				'out_mean': tf.Variable(xavier_init(n_in, n_z),name='embmap_out_mean')	
			}
			all_weights['embmap_biases']={
				'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32),name='embmap_out_meanb',trainable=embeddings_trainable)
			}
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
		if not form3:
			z,vae_loss=self._vae_sample(ve_weights[0],ve_weights[1],self.x)
		else:
			with tf.device('/cpu:0'):
				x=tf.nn.embedding_lookup(self.embw,self.x)
			x+=self.embb
			z,vae_loss=self._vae_sample_mid(ve_weights[0],ve_weights[1],x)
		self.word_var=z
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
		print z.shape
		self.mid_var=z
		embedding=tf.matmul(z,lstm_weights['affine_weight'])+lstm_weights['affine_bias']
		return embedding,vae_loss

	def _get_word_embedding(self, ve_weights, lstm_weights, x,logit=False):
		if form3:
			with tf.device('/cpu:0'):
				x=tf.nn.embedding_lookup(self.embw,self.x)
			x+=self.embb
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
				print 'stop fucking sampling',mid_vae
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
		
	def partial_fit(self, X,y,mask,testify=False,timestep=0):
		"""Train model based on mini-batch of input data.
		
		Return cost of mini-batch.
		"""
		if self.test and testify:
			print tf.test.compute_gradient_error(self.x,np.array([self.batch_size,self.n_words]),self.loss,[self.batch_size],extra_feed_dict={self.caption_placeholder: y, self.mask: mask})
			exit()
		else:
			opt, cost,shit = self.sess.run((self.optimizer, self.loss,self.debug), 
									  feed_dict={self.x: X, self.caption_placeholder: y, self.mask: mask,self.timestep:timestep})
			# print shit
			# print deb
			# exit()
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
			input_embedding,_=self._get_input_embedding([network_weights['embmap'],network_weights['embmap_biases']],network_weights['embmap'])
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
			if form4:
				previous_word,_=input_embedding,None
			elif form2:
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
					best_word = tf.argmax(logit, 1)

				# with tf.device("/cpu:0"):
				#     # get the embedding of the best_word to use as input to the next iteration of our LSTM 
				#     previous_word = tf.nn.embedding_lookup(self.word_embedding, best_word)

				# previous_word += self.embedding_bias
				print logit.shape
				if form4:
					previous_word,_=input_embedding,None
				elif form2:
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
	global_step=tf.Variable(0,trainable=False)
	total_batch = int(n_samples / batch_size)
	if should_decay and not gen:
		
		learning_rate = tf.train.exponential_decay(learning_rate, global_step,
                                           total_batch, 0.95, staircase=True)
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
	# indlist=np.arange(10*batch_size).astype(int)
	for epoch in range(training_epochs):
		avg_cost = 0.
		
		# Loop over all batches
		
		np.random.shuffle(indlist)
		testify=False
		avg_loss=0
		# for i in range(1):
		for i in range(total_batch):
			# break
			ts=i
			# i=0
			inds=np.random.choice(indlist,batch_size)
			# print indlist[i*batch_size:(i+1)*batch_size]
			# batch_xs = X[indlist[i*batch_size:(i+1)*batch_size]]
			batch_xs = X[inds]

			# Fit training using batch data
			# if epoch==2 and i ==0:
			# 	testify=True
			# cost,loss = vae.partial_fit(batch_xs,y[indlist[i*batch_size:(i+1)*batch_size]].astype(np.uint32),mask[indlist[i*batch_size:(i+1)*batch_size]],timestep=epoch*total_batch+ts,testify=testify)
			cost,loss = vae.partial_fit(batch_xs,y[inds].astype(np.uint32),mask[inds],timestep=(epoch)+1e-3,testify=testify)

			# Compute average loss
			avg_cost = avg_cost * i /(i+1) +cost/(i+1)
			# avg_loss=avg_loss*i/(i+1)+loss/(i+1)
			if i% display_step==0:
				print avg_cost,loss,cost
			if epoch == 0 and ts==0:
				costs.append(avg_cost)
			

		
		costs.append(avg_cost)
		
		# Display logs per epoch step
		if epoch % (display_step*10) == 0 or epoch==1:
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
	form3=	True
	form4=False
	vanilla=True

	if sys.argv[2]=='mid_vae':
		mid_vae=True
		print 'mid_vae'
	same_embedding=False
	clip_grad=True
	if sys.argv[3]!='clip':
		clip_grad=False
	should_save=True
	should_train=True
	# should_train=not should_train
	should_continue=False
	# should_continue=True
	should_decay=True
	zero_end_tok=True
	training_epochs=int(sys.argv[13])
	batch_size=int(sys.argv[4])
	onehot=False
	embeddings_trainable=False
	if sys.argv[5]!='transfer':
		print 'true embs'
		embeddings_trainable=True
	transfertype2=True
	binary_dim=int(sys.argv[6])
	all_the_f_one_h=[]
	if not zero_end_tok:
		X, y, mask, _map = load_text(2**binary_dim-4)
	else:
		X, y, mask, _map = load_text(150000)
	n_input =150000
	n_samples = 30000
	lstm_dim=int(sys.argv[7])
	model_path = sys.argv[8]
	vartype=''
	transfertype=''
	maxlen=int(sys.argv[9])+2
	n_z=int(sys.argv[10])
	n_z_m=int(sys.argv[11])
	n_z_m_2=int(sys.argv[12])
	if not vanilla:
		vartype='var'
	if not embeddings_trainable:
		transfertype='transfer'
	cliptype=''
	if clip_grad:
		cliptype='clip'
	use_ctc=False
	losstype=''
	if sys.argv[14]=='ctc_loss':
		use_ctc=True
		losstype='ctc'
	lstm_stack=int(sys.argv[15])
	use_bdlstm=False
	bdlstmtype=''
	if sys.argv[16]!='forward':
		use_bdlstm=True
		bdlstmtype='bdlstm'
	loss_output_path= 'losses/%s%ss_%sb_%sl_%sh_%sd_%sz_%szm_%s%s%sdefdef%s4.pkl'%(bdlstmtype,str(lstm_stack),str(batch_size),str(maxlen-2),str(lstm_dim),str(n_input),str(n_z),str(n_z_m),str(losstype),str(cliptype),str(vartype),str(transfertype))
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

	# batch_size=1
	if should_train:
		# vae_2d = train(network_architecture, training_epochs=training_epochs, batch_size=batch_size,gen=False,ctrain=should_continue)
		# print train(network_architecture, training_epochs=training_epochs, batch_size=batch_size,gen=False,ctrain=should_continue,test=True)
		vae_2d = train(network_architecture, training_epochs=training_epochs, batch_size=batch_size,gen=False,ctrain=should_continue,learning_rate=.005)
	else:
		vae_2d = train(network_architecture, training_epochs=training_epochs, batch_size=batch_size,gen=True,ctrain=True)
	
	# #	vae_2d._build_gen()
		ind_list=np.arange(len(X)).astype(int)
		# np.random.shuffle(ind_list)
		x_sample = X[ind_list[:batch_size]]
		print x_sample
		y_sample = y[ind_list[:batch_size]]
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
		print(ixtoword(_map,bin_to_int(np.expand_dims(x_sample[:10],axis=0))))
	# 	# plt.figure(figsize=(8, 6)) 
		# plt.scatter(z_mu[:, 0], z_mu[:, 1], c=np.argmax(y_sample, 1))
		# plt.colorbar()
		# plt.grid()
		# plt.show()
