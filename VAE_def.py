''' Significant lifting from https://jmetzen.github.io/2015-11-27/vae.html '''
import time
#time.sleep(3600) LOL why
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

import re, string
from sklearn.feature_extraction.text import CountVectorizer

from collections import defaultdict
import pickle as pkl


def load_text(n,num_samples=None):
	fname = 'Oxford_English_Dictionary.txt'
	txt = []
	with open(fname,'rb') as f:
		txt = f.readlines()

	txt = [x.decode('utf-8').strip() for x in txt]
	txt = [re.sub(r'[^a-zA-Z ]+', '', x) for x in txt if len(x) > 1]

	# List of words
	word_list = [x.split(' ', 1)[0].strip() for x in txt]
	# List of definitions
	def_list = [x.split(' ', 1)[1].strip()for x in txt]

	maxlen=0
	for defi in def_list:
		maxlen=max(maxlen,len(defi.split()))
	print (maxlen)
	maxlen=30

	# # Initialize the "CountVectorizer" object, which is scikit-learn's
	# # bag of words tool.  
	# vectorizer = CountVectorizer(analyzer = "word",   \
	#                              tokenizer = None,    \
	#                              preprocessor = None, \
	#                              stop_words = None,   \
	#                              max_features = None, \
	#                              token_pattern='\\b\\w+\\b') # Keep single character words

	_map,rev_map=get_one_hot_map(word_list+def_list,n)
	if num_samples is not None:
		num_samples=len(word_list)
	# X = (36665, 56210)

	X = map_one_hot(word_list[:num_samples],_map,1,n)
	# y = (36665, 56210)
	y,mask = map_one_hot(def_list[:num_samples],_map,maxlen,n)
	print (np.max(y))
	return X, y, mask,rev_map

def get_one_hot_map(corpus,n):
	counts=defaultdict(int)
	for line in corpus:
		for word in line.split():
			counts[word]+=1
	_map=defaultdict(lambda :n+1)
	rev_map={}
	words=list(sorted(counts.items(),key=lambda x:x[1]))[:n]
	i=0
	for word,count in words:
		i+=1
		_map[word]=i
		rev_map[i]=word
	rev_map[n+1]='<UNK>'
	rev_map[0]='Start'
	rev_map[n+2]='End'
	return _map,rev_map

def map_one_hot(corpus,_map,maxlen,n):
	if maxlen==1:
		rtn=np.zeros([len(corpus),n+3],dtype=np.float32)
		print (len(corpus))
		print (len(_map.items())+2)
		for l,line in enumerate(corpus):
			if len(line)==0:
				rtn[l,-1]=1
			else:
				
				rtn[l,_map[line]]=1
		return rtn
	else:
		rtn=np.zeros([len(corpus),maxlen+2],dtype=np.int64)
		print (rtn.shape)
		mask=np.zeros([len(corpus),maxlen+2],dtype=np.float32)
		print (mask.shape)
		mask[:,1]=1.0
		for l,_line in enumerate(corpus):
			x=-1
			line=_line.split()
			for i in range(min(len(line),maxlen)):
				rtn[l,i+1]=_map[line[i]]
				mask[l,i+1]=1.0
				x=i
			rtn[l,x+2]=n+2
		return rtn,mask


def xavier_init(fan_in, fan_out, constant=1): 
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
				 learning_rate=0.001, batch_size=100):
		self.network_architecture = network_architecture
		self.transfer_fct = transfer_fct
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		
		# tf Graph input
		self.n_words=network_architecture['n_input']
		self.x = tf.placeholder(tf.float32, [None,self.n_words])
		self.intype=type(self.x)
		self.caption_placeholder = tf.placeholder(tf.int32, [None, network_architecture["maxlen"]])
		self.mask=tf.placeholder(tf.float32, [None, network_architecture["maxlen"]])
		
		# Create autoencoder network
		self._create_network()
		# Define loss function based variational upper-bound and 
		# corresponding optimizer
		self._create_loss_optimizer()
		
		# Initializing the tensor flow variables

		init = tf.global_variables_initializer()

		# Launch the session
		self.sess = tf.InteractiveSession()
		self.saver = tf.train.Saver(max_to_keep=100)
		self.sess.run(init)
	
	def _create_network(self):
		# Initialize autoencode network weights and biases
		network_weights = self._initialize_weights(**self.network_architecture)
		self.network_weights=network_weights
		# Use recognition network to determine mean and 
		# (log) variance of Gaussian distribution in latent
		# space
		input_embedding,input_embedding_KLD_loss=self._get_input_embedding([network_weights['variational_encoding'],network_weights['biases_variational_encoding']],network_weights['input_meaning'])

		state = self.lstm.zero_state(self.batch_size, dtype=tf.float32)

		loss = input_embedding_KLD_loss
		with tf.variable_scope("RNN"):
			for i in range(self.network_architecture['maxlen']): 
				if i > 0:
					with tf.device("/cpu:0"):
						# current_embedding = tf.nn.embedding_lookup(self.word_embedding, caption_placeholder[:,i-1]) + self.embedding_bias
						current_embedding,KLD_loss = self._get_word_embedding([network_weights['variational_encoding'],network_weights['biases_variational_encoding']],network_weights['LSTM'], self.caption_placeholder[:,i-1])
						loss+=KLD_loss
				else:
					 current_embedding = input_embedding
				if i > 0: 
					tf.get_variable_scope().reuse_variables()

				out, state = self.lstm(current_embedding, state)

				
				if i > 0: 
					labels = tf.expand_dims(self.caption_placeholder[:, i], 1)
					ix_range=tf.range(0, self.batch_size, 1)
					ixs = tf.expand_dims(ix_range, 1)
					concat = tf.concat([ixs, labels],1)
					onehot = tf.sparse_to_dense(
							concat, tf.stack([self.batch_size, self.n_words]), 1.0, 0.0)

					logit = tf.matmul(out, network_weights['LSTM']['encoding_weight']) + network_weights['LSTM']['encoding_bias']
					xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=onehot)
					xentropy = xentropy * self.mask[:,i]

					loss += tf.reduce_sum(xentropy)

			loss = loss / tf.reduce_sum(self.mask[:,1:])
			print (loss.shape)
			self.loss=loss
	
	def _initialize_weights(self, n_lstm_input, maxlen, 
							n_input, n_z):
		all_weights = dict()
		all_weights['input_meaning'] = {
			'affine_weight': tf.Variable(xavier_init(n_z, n_lstm_input)),
			'affine_bias': tf.Variable(tf.zeros(n_lstm_input))}
		all_weights['biases_variational_encoding'] = {
			'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
			'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32))}
		all_weights['variational_encoding'] = {
			'out_mean': tf.Variable(xavier_init(n_input, n_z)),
			'out_log_sigma': tf.Variable(xavier_init(n_input, n_z))}
		self.lstm=tf.contrib.rnn.BasicLSTMCell(n_lstm_input)
		all_weights['LSTM'] = {
			'affine_weight': tf.Variable(xavier_init(n_z, n_lstm_input)),
			'affine_bias': tf.Variable(tf.zeros(n_lstm_input)),
			'encoding_weight': tf.Variable(xavier_init(n_lstm_input,n_input)),
			'encoding_bias': tf.Variable(tf.zeros(n_input)),
			'lstm': self.lstm}
		return all_weights
	
	def _get_input_embedding(self, ve_weights, aff_weights):
		z,vae_loss=self._vae_sample(ve_weights[0],ve_weights[1],self.x)
		embedding=tf.matmul(z,aff_weights['affine_weight'])+aff_weights['affine_bias']
		return embedding,vae_loss

	def _get_word_embedding(self, ve_weights, lstm_weights, x):
		z,vae_loss=self._vae_sample(ve_weights[0],ve_weights[1],x, True)
		embedding=tf.matmul(z,lstm_weights['affine_weight'])+lstm_weights['affine_bias']
		return embedding,vae_loss
	

	def _vae_sample(self, weights, biases, x, lookup=False):
			#TODO: consider adding a linear transform layer+relu or softplus here first 
			if not lookup:
				mu=tf.matmul(x,weights['out_mean'])+biases['out_mean']
				logvar=tf.matmul(x,weights['out_log_sigma'])+biases['out_log_sigma']
			else:
				mu=tf.nn.embedding_lookup(weights['out_mean'],x)+biases['out_mean']
				logvar=tf.nn.embedding_lookup(weights['out_log_sigma'],x)+biases['out_log_sigma']
			epsilon=tf.random_normal(tf.shape(logvar),name='epsilon')
			std=tf.exp(.5*logvar)
			z=mu+tf.multiply(std,epsilon)
			KLD = -0.5 * tf.reduce_sum(1 + logvar - tf.pow(mu, 2) - tf.exp(logvar), reduction_indices=1)
			return z,KLD

	def _create_loss_optimizer(self):
		self.optimizer = \
			tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
		
	def partial_fit(self, X,y,mask):
		"""Train model based on mini-batch of input data.
		
		Return cost of mini-batch.
		"""
		opt, cost = self.sess.run((self.optimizer, self.loss), 
								  feed_dict={self.x: X, self.caption_placeholder: y, self.mask: mask})
		return cost
		
	def _build_gen(self):
		#same setup as `_create_network` function 
		input_embedding,_=self._get_input_embedding([self.network_weights['variational_encoding'],self.network_weights['biases_variational_encoding']],self.network_weights['input_meaning'])
        # image_embedding = tf.matmul(img, self.img_embedding) + self.img_embedding_bias
        state = self.lstm.zero_state(self.batch_size,dtype=tf.float32)

        #declare list to hold the words of our generated captions
        all_words = []
        with tf.variable_scope("RNN"):
            # in the first iteration we have no previous word, so we directly pass in the image embedding
            # and set the `previous_word` to the embedding of the start token ([0]) for the future iterations
            output, state = self.lstm(image_embedding, state)
            previous_word,_ = self._get_word_embedding([network_weights['variational_encoding'],network_weights['biases_variational_encoding']],network_weights['LSTM'], [0])
            # previous_word = tf.nn.embedding_lookup(self.word_embedding, [0]) + self.embedding_bias

            for i in range(self.network_architecture['maxlen']):
                tf.get_variable_scope().reuse_variables()

                out, state = self.lstm(previous_word, state)


                # get a one-hot word encoding from the output of the LSTM
                logit = tf.matmul(out, self.word_encoding) + self.word_encoding_bias
                best_word = tf.argmax(logit, 1)

                # with tf.device("/cpu:0"):
                #     # get the embedding of the best_word to use as input to the next iteration of our LSTM 
                #     previous_word = tf.nn.embedding_lookup(self.word_embedding, best_word)

                # previous_word += self.embedding_bias

                previous_word,_ = self._get_word_embedding([network_weights['variational_encoding'],network_weights['biases_variational_encoding']],network_weights['LSTM'], [best_word])

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
	    
	    saver = tf.train.Saver()
	    saver.restore(sess, tf.train.latest_checkpoint(model_path))

	    generated_word_index= sess.run(self.generated_words, feed_dict={self.x:x})
	    generated_word_index = np.hstack(generated_word_index)

	    generated_sentence = ixtoword(_map,generated_word_index)
	    return generated_sentence

def ixtoword(_map,ixs):
	return [_map[x] for x in ixs]
	

def train(network_architecture, learning_rate=0.001,
		  batch_size=100, training_epochs=10, display_step=5):
	vae = VariationalAutoencoder(network_architecture, 
								 learning_rate=learning_rate, 
								 batch_size=batch_size)
	# Training cycle
	costs=[]
	for epoch in range(training_epochs):
		avg_cost = 0.
		total_batch = int(n_samples / batch_size)
		# Loop over all batches
		for i in range(total_batch):
			batch_xs = X[i*n_samples:n_samples, :]

			# Fit training using batch data
			cost = vae.partial_fit(batch_xs,y[i*n_samples:n_samples,:],mask[i*n_samples:n_samples,:])
			# Compute average loss
			avg_cost += np.sum(cost) / n_samples * batch_size

		vae.saver.save(vae.sess, './models/model')
		costs.append(avg_cost)
		pkl.dump(costs,open('50_256_results.pkl','wb'))
		# Display logs per epoch step
		if epoch % display_step == 0:
			print("Epoch:", '%04d' % (epoch+1), 
				  "cost=", avg_cost)
	return vae

def main():

	X, y, mask, _map = load_text(1000,1000)

	n_input = 1003
	n_samples = 500
	lstm_dim=256

	X, y = X[:n_samples, :], y[:n_samples, :]

	network_architecture = \
		dict(maxlen=32, # 2nd layer decoder neurons
			 n_input=n_input, # One hot encoding input
			 n_lstm_input=lstm_dim, # LSTM cell size
			 n_z=50, # dimensionality of latent space
			 )  

	vae_2d = train(network_architecture, training_epochs=300, batch_size=500)

	x_sample = X[:10]
	y_sample = y[:10]
	y_hat = vae_2d.generate(_map,x_sample)
	y_hat_words=ixtoword(_map,y_hat)
	y_words=ixtoword(_map,y_sample)
	# plt.figure(figsize=(8, 6)) 
	# plt.scatter(z_mu[:, 0], z_mu[:, 1], c=np.argmax(y_sample, 1))
	# plt.colorbar()
	# plt.grid()
	# plt.show()

if __name__ == "__main__":
	main()