import tensorflow as tf
from tensorflow.python.ops import rnn
def linear(input_, output_size, scope=None):
    '''
    Linear map: output[k] = sum_i(Matrix[k, i] * input_[i] ) + Bias[k]
    Args:
    input_: a tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    scope: VariableScope for the created subgraph; defaults to "Linear".
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(input_[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  '''

    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "SimpleLinear"):
        matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
        bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)

    return tf.matmul(input_, tf.transpose(matrix)) + bias_term

def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """

    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = f(linear(input_, size, scope='highway_lin_%d' % idx))

            t = tf.sigmoid(linear(input_, size, scope='highway_gate_%d' % idx) + bias)

            output = t * g + (1. - t) * input_
            input_ = output

    return output
def xavier_init(fan_in, fan_out, constant=1e-4): 
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), 
                             minval=low, maxval=high, 
                             dtype=tf.float32)
class DC(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(
            self, sequence_length, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0,emb_dim_in=None):
        # Placeholders for input, output and dropout
        if emb_dim_in is None:
            self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        else:
            self.input_x = tf.placeholder(tf.float32, [None, sequence_length,emb_dim_in], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
        
        with tf.variable_scope('discriminator'):

            # Embedding layer
            if emb_dim_in is None:
                with tf.device('/cpu:0'), tf.name_scope("embedding"):
                    self.W = tf.Variable(
                        tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                        name="W")
                    self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
                self.b=tf.Variable(tf.zeros([embedding_size]),dtype=tf.float32)
                self.embedded_chars+=self.b
                self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            else:
                self.W = tf.Variable(
                    tf.random_uniform([emb_dim_in, embedding_size], -1.0, 1.0),
                    name="W")
                self.embedded_chars = tf.matmul(tf.reshape(self.input_x,[-1,emb_dim_in]),self.W)
                self.b=tf.Variable(tf.zeros([embedding_size]),dtype=tf.float32)
                self.embedded_chars+=self.b
                self.embedded_chars=tf.reshape(self.embedded_chars,[-1,sequence_length,embedding_size])
                self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            for filter_size, num_filter in zip(filter_sizes, num_filters):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, embedding_size, 1, num_filter]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filter]), name="b")
                    conv = tf.nn.conv2d(
                        self.embedded_chars_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)
            
            # Combine all the pooled features
            num_filters_total = sum(num_filters)
            self.h_pool = tf.concat(pooled_outputs, 3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

            # Add highway
            with tf.name_scope("highway"):
                self.h_highway = highway(self.h_pool_flat, self.h_pool_flat.get_shape()[1], 1, 0)

            # Add dropout
            with tf.name_scope("dropout"):
                self.h_drop = tf.nn.dropout(self.h_highway, self.dropout_keep_prob)

            # Final (unnormalized) scores and predictions
            with tf.name_scope("output"):
                W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
                self.ypred_for_auc = tf.nn.softmax(self.scores)
                self.predictions = tf.argmax(self.scores, 1, name="predictions")

            # CalculateMean cross-entropy loss
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
                self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        self.params = [param for param in tf.trainable_variables() if 'discriminator' in param.name]
        d_optimizer = tf.train.AdamOptimizer(1e-4)
        grads_and_vars = d_optimizer.compute_gradients(self.loss, self.params, aggregation_method=2)
        self.train_op = d_optimizer.apply_gradients(grads_and_vars)

class DLSTM(object):
    def __init__(self,sequence_length,num_classes,hidden_dim,middle_dim,emb_dim_in,bidir=False,):
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        
        # self.vl=[self.W,self.b,self.W2,self.b2]
        self.forward_cell=tf.contrib.rnn.BasicLSTMCell(hidden_dim)
        if bidir:
            self.backward_cell=tf.contrib.rnn.BasicLSTMCell(hidden_dim)
        self.bidir=bidir
        self.num_classes=num_classes
        self.hidden_dim=hidden_dim
        self.sequence_length=sequence_length
        self.middle_dim=middle_dim
        self.emb_dim_in=emb_dim_in
        trainability=False
        # with tf.device('/cpu:0'):
        #     om=tf.Variable(xavier_init(self.n_z, self.n_z),name='out_mean',trainable=trainability)
        # if not vanilla:
        #     all_weights['biases_variational_encoding'] = {
        #         'out_mean': tf.Variable(tf.zeros([self.n_z], dtype=tf.float32),name='out_meanb',trainable=trainability),
        #         'out_log_sigma': tf.Variable(tf.zeros([self.n_z], dtype=tf.float32),name='out_log_sigmab',trainable=trainability)}
        #     all_weights['variational_encoding'] = {
        #         'out_mean': om,
        #         'out_log_sigma': tf.Variable(xavier_init(self.n_input, self.n_z),name='out_log_sigma',trainable=trainability)}
            
        # else:
        #     all_weights['biases_variational_encoding'] = {
        #         'out_mean': tf.Variable(tf.zeros([self.n_z], dtype=tf.float32),name='out_meanb',trainable=trainability)}
        #     all_weights['variational_encoding'] = {
        #         'out_mean': om}
        self.W=None
        if not self.bidir:
            self.W=tf.Variable(tf.random_normal([self.hidden_dim*self.sequence_length,self.middle_dim]),name='dlstm')
        else:
            self.W=tf.Variable(tf.random_normal([2*self.hidden_dim*self.sequence_length,self.middle_dim]),name='dlstm')

        self.b=tf.Variable(tf.zeros([self.middle_dim]),name='dlstmb')
        self.W2=tf.Variable(tf.random_normal([self.middle_dim,self.num_classes]),name='dlstm2')
        self.b2=tf.Variable(tf.zeros([self.num_classes]),name='dlstm2b')
        # all_encoding_weights=[all_weights[x].values() for x in all_weights]
        # encoding_weights=[]
        # for w in all_encoding_weights:
        #     encoding_weights+=w
        # self.Dvars=[self.W,self.b,self.W2,self.b2]+encoding_weights
        # self._build_network()
    def _build_network(self):
        
        embedded_input,KLD_loss=self._get_word_embedding([all_weights['variational_encoding'],all_weights['biases_variational_encoding']],None,tf.reshape(self.input_x,[-1]),logit=True)
        embedded_input=tf.reshape(embedded_input,[-1,self.sequence_length,self.emb_dim_in])
        if self.bidir:
            outs,_,_=tf.contrib.rnn.static_bidirectional_rnn(self.forward_cell,self.backward_cell,embedded_input)
            outs=tf.reshape(outs,[-1,self.sequence_length*2*self.hidden_dim])
        else:
            outs,states=tf.contrib.rnn.static_rnn(self.forward_cell,embedded_input)
            
            outs=tf.reshape(outs,[-1,self.sequence_length*self.hidden_dim])

        
        middle=tf.matmul(outs,self.W)+self.b
        middle=tf.nn.dropout(self.dropout_keep_prob)
        
        out=tf.matmul(middle,self.W2)+self.b2
        self.D1=tf.sigmoid(out)
    def discriminate(self,input_x,train=True):
        if self.bidir:
            outs,_,_=tf.contrib.rnn.static_bidirectional_rnn(self.forward_cell,self.backward_cell,input_x,time_major=False)
            outs=tf.reshape(outs,[-1,self.sequence_length*2*self.hidden_dim])
        else:
            outs,states=rnn.dynamic_rnn(self.forward_cell,input_x,dtype=tf.float32,time_major=False)
            outs=tf.reshape(outs,[-1,self.sequence_length*self.hidden_dim])
        
        middle=tf.matmul(outs,self.W)+self.b
        middle=tf.nn.dropout(middle,self.dropout_keep_prob)
        out=tf.matmul(middle,self.W2)+self.b2
        return tf.sigmoid(out)
        # return loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out,labels=input_y))
    