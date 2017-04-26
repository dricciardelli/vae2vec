import tensorflow as tf

def get_output_probabilities(output_probabilities,captions):
	pre_prob=tf.pow(output_probabilities,captions)
	pre_prob=1-captions+(2*captions-1)*output_probabilities
	pre_prob=tf.maximum(pre_prob,tf.constant(1e-3,dtype=tf.float32))
	pre_prob=tf.minimum(pre_prob,tf.constant(1.0-1e-3,dtype=tf.float32))
	return tf.reduce_prod(pre_prob,axis=2)
def loss(output_probabilities,captions,timesteps,batch_size,seqlen):
	assignment_position=1
	predictions=[]
	indices=[]
	labels=[]
	# values=tf.ones([batch_size*timesteps],dtype=tf.int32)
	indices=[]
	for i in range(batch_size):
		ixs=tf.expand_dims(tf.range(0,seqlen[i],1),-1)
		batch_ind=tf.ones([seqlen[i],1],dtype=tf.int32)*i
		indices.append(tf.concat([batch_ind,ixs],axis=-1))
	indices=tf.cast(tf.concat(indices,axis=0),dtype=tf.int64)
	# values=tf.ones([indices.shape[0]],dtype=tf.int32)
	values=tf.ones([tf.reduce_sum(seqlen)],dtype=tf.int32)
	for i in range((timesteps)):
		# if i!=0:
			# different=1-tf.reduce_prod(tf.cast(tf.equal(captions[:,i],captions[:,i-1]),tf.float32),axis=-1)
			# assignment_position=assignment_position*(1-different)+(1-assignment_position)*different
		correct_prob=output_probabilities[:,i]
		incorrect_prob=1-output_probabilities[:,i]

		prediction=tf.concat([tf.expand_dims(incorrect_prob,-1),tf.expand_dims(correct_prob,-1)],axis=-1)
		prediction=-tf.log(prediction)
		predictions.append(tf.expand_dims(prediction,axis=0))

		
	print indices.shape
	targets=tf.SparseTensor(indices,values,tf.constant([batch_size,timesteps],dtype=tf.int64))
	predictions=tf.concat(predictions,axis=0)
	print predictions.shape
	# predictions=tf.transpose(predictions,[1,0,2])
	print predictions.shape
	# print targets.shape
	loss=tf.nn.ctc_loss(targets,predictions,sequence_length=tf.cast(seqlen,tf.int32),ctc_merge_repeated=False)
	return tf.reduce_mean(loss)
