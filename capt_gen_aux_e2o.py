# -*- coding: utf-8 -*-
import math
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import pickle as pkl
import cv2
import skimage
import random
import tensorflow.python.platform
from tensorflow.python.ops import rnn
from keras.preprocessing import sequence
from collections import Counter
from collections import defaultdict
import itertools
test_image_path='./data/acoustic-guitar-player.jpg'
vgg_path='./data/vgg16-20160129.tfmodel'
n=50000-2
def map_lambda():
    return n+1
def rev_map_lambda():
    return "<UNK>"
def load_text(n,capts,num_samples=None):
    # fname = 'Oxford_English_Dictionary.txt'

    # txt = []
    # with open(fname,'rb') as f:
    #   txt = f.readlines()

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
    wd={}
    while i<len( dl):
        defi=dl[i]
        if len(defi)>0:
            def_list+=[' '.join(defi).lower()]
            i+=1
            word=word_list[i-1].lower()
            word_list[i-1]=word
            if word not in wd:
                wd[word]=[]
            wd[word].append(def_list[-1])
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

    _map,rev_map=get_one_hot_map(word_list,def_list,n,captlist=capts)
    # pkl.dump(_map,open('mapaoh.pkl','wb'))
    # pkl.dump(rev_map,open('rev_mapaoh.pkl','wb'))
    _map=pkl.load(open('mapaoh.pkl','rb'))
    rev_map=pkl.load(open('rev_mapaoh.pkl','rb'))
    if num_samples is not None:
        num_samples=len(capts)
    # X = map_one_hot(word_list[:num_samples],_map,1,n)
    # y = (36665, 56210)
    # print _map
    if capts is not None:
        
        # y,mask,auxsent,auxmask,auxword,auxchoices = map_one_hot(capts[:num_samples],_map,maxlen,n,aux=True,wd=wd)
        # np.save('maskmainaux',mask)
        # np.save('ycoh',y)
        # np.save('yaux',auxsent)
        # np.save('maskaux',auxmask)
        # np.save('Xaux',auxword)
        # np.save('caux',auxchoices)
        # exit()
        print capts
        y=np.load('ycoh.npy')#,'r')
        auxmask=np.load('maskaux.npy')#,'r')
        mask=np.load('maskmainaux.npy')#,'r')
        auxword=np.load('Xaux.npy')#,'r')
        auxsent=np.load('yaux.npy')#,'r')
        auxchoices=np.load('caux.npy')#,'r')
    else:
        # np.save('X',X)
        # np.save('yc',y)
        # np.save('maskc',mask)
        mask=np.load('maskaoh.npy','r')
        y=np.load('yaoh.npy','r')
    X=np.load('Xaoh.npy')#,'r')
    
    print (np.max(y))
    if capts is not None:
        return X, y,mask,rev_map,auxsent,auxmask,auxword,auxchoices
    return X, y, mask,rev_map

def get_one_hot_map(to_def,corpus,n,captlist=None):
    # words={}
    # for line in to_def:
    #   if line:
    #       words[line.split()[0]]=1

    

    # counts=defaultdict(int)
    # uniq=defaultdict(int)
    # for line in corpus:
    #   for word in line.split():
    #       if word not in words:
    #           counts[word]+=1
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
    # print (len(words))
    counts2=defaultdict(int)
    if captlist is not None:
        for line in captlist:
            for word in line.split():
                if word=='#START#':
                    continue
                counts2[word.lower()]+=1
    print len(counts.keys()),len(counts2.keys())
    words=list(map(lambda z:z[0],reversed(sorted(counts2.items(),key=lambda x:x[1]))))[:n-len(words)]
    # random.shuffle(words)
    words=words[:3000]
    for word in words:
        if word in counts:
            del counts[word]
    print len(counts.keys()),len(counts2.keys())
    words+=list(map(lambda z:z[0],reversed(sorted(counts.items(),key=lambda x:x[1]))))[:n-len(words)]
    print (len(words))
    i=0
    # random.shuffle(words)
    # for num_bits in range(binary_dim):
    #     for bit_config in itertools.combinations_with_replacement(range(binary_dim),num_bits+1):
    #         bitmap=np.zeros(binary_dim)
    #         bitmap[np.array(bit_config)]=1
    #         num=bitmap*(2** np.arange(binary_dim ))
    #         num=np.sum(num).astype(np.uint32)
    #         word=words[i]
    #         _map[word]=num
    #         rev_map[num]=word
    #         i+=1
    #         if i>=len(words):
    #             break
    #     if i>=len(words):
    #             break
    _map['#START#']=0
    for word in words:
        i+=1
        _map[word]=i
        rev_map[i]=word
    rev_map[n+1]='<UNK>'
    if zero_end_tok:
        rev_map[0]='.'
    else:
        rev_map[0]='Start'
        rev_map[n+2]='End'
    # print (list(reversed(sorted(uniq.items()))))
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

def map_one_hot(corpus,_map,maxlen,n,aux=None,wd=None):
    if maxlen==1:
        if not form2:
            total_not=0
            rtn=np.zeros([len(corpus),n+3],dtype=np.float32)
            for l,_line in enumerate(corpus):
                line=_line.lower()
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
            for l,_line in enumerate(corpus):
                line=_line.lower()
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
        rtn3=[]
        rtn3=np.zeros([len(corpus),5,maxlen+2],dtype=np.int32)
        rtn4=[]
        rtn4=np.zeros([len(corpus),5,maxlen+2],dtype=np.float32)
        rtn5=[]
        rtn5=np.zeros([len(corpus),5,1],dtype=np.int32)
        rtn6=[]
        rtn6=np.zeros([len(corpus),5,1],dtype=np.float32)
        for l,_line in enumerate(corpus):
            x=0
            line=_line.lower().split()
            auxlist=[]
            auxmask=[]
            auxword=[]
            auxchoices=[]
            for i in range(min(len(line),maxlen-1)):
                # if line[i] not in _map:
                #   nopes+=1

                mapped=_map[line[i]]
                rtn[l,i+1]=mapped
                y=0
                if not (wd is None) and mapped!=n+1 and line[i] in wd:
                    tempsent=np.zeros([1,maxlen+2],dtype=np.int32)
                    _sent=random.choice(wd[line[i]])
                    tempmask=np.zeros([1,maxlen+2],dtype=np.float32)
                    tempword=np.ones([1,1],dtype=np.int32)
                    tempmask[0,1]=1.0
                    tempword*=mapped
                    tempchoice=np.ones([1,1],dtype=np.float32)
                    sent=_sent.split()
                    for j in range(min(len(sent),maxlen-1)):
                        m2=_map[sent[j]]
                        tempsent[0,j+1]=m2
                        tempmask[0,j+1]=1.0
                        y=j+1
                    tempsent[0,y]=0
                    tempmask[0,y]=1.0
                    auxlist.append(tempsent)
                    auxmask.append(tempmask)
                    auxword.append(tempword)
                    auxchoices.append(tempchoice)
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
            ilist=np.arange(len(auxlist))
            if len(auxlist)>=5:
                random.shuffle(ilist)
                auxlist=np.concatenate(auxlist) 
                auxlist=auxlist[ilist[:5]]
            elif len(auxlist)>0:
                # print auxlist
                # print [x.shape for x in auxlist]
                auxlist+=[np.zeros([5-len(auxlist),maxlen+2],dtype=np.int32)]
                # print auxlist
                # print [x.shape for x in auxlist]
                auxlist=np.concatenate(auxlist)
            else:
                auxlist=np.zeros([5,maxlen+2],dtype=np.int32)
            # rtn3.append(auxlist)
            rtn3[l,:,:]=auxlist
            # print rtn3
            if len(auxmask)>=5:
                auxmask=np.concatenate(auxmask) 
                auxmask=auxmask[ilist[:5]]
            elif len(auxmask)>0:
                
                auxmask+=[np.zeros([5-len(auxmask),maxlen+2],dtype=np.float32)]
                auxmask=np.concatenate(auxmask)
            else:
                auxmask=np.zeros([5,maxlen+2],dtype=np.float32)
            # rtn4.append(auxmask)
            # print l
            rtn4[l,:,:]=auxmask
            if len(auxword)>=5:
                auxword=np.concatenate(auxword) 
                auxword=auxword[ilist[:5]]
            elif len(auxlist)>0:
                
                auxword+=[np.zeros([5-len(auxword),1],dtype=np.int32)]
                auxword=np.concatenate(auxword)
            else:
                auxword=np.zeros([5,1],dtype=np.int32)
            # rtn5.append(auxword)
            rtn5[l,:,:]=auxword
            if len(auxchoices)>=5:
                auxchoices=np.concatenate(auxchoices) 
                auxchoices=auxchoices[ilist[:5]]
            elif len(auxlist)>0:
                
                auxchoices+=[np.zeros([5-len(auxchoices),1],dtype=np.float32)]
                auxchoices=np.concatenate(auxchoices)
            else:
                auxchoices=np.zeros([5,1],dtype=np.float32)
            # rtn6.append(auxchoices)
            rtn6[l,:,:]=auxchoices

        print (nopes,totes,wtf)
        if not (aux is None):
            # print np.array(rtn6)[-1],np.array(rtn4)[-1]
            # return rtn,mask,np.array(rtn3),np.array(rtn4),np.array(rtn5),np.array(rtn6)
            return rtn,mask,rtn3,rtn4,rtn5,rtn6
        else:
            return rtn,mask


def xavier_init(fan_in, fan_out, constant=1e-4): 
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), 
                             minval=low, maxval=high, 
                             dtype=tf.float32)

class Caption_Generator():
    def __init__(self, dim_in, dim_embed, dim_hidden, batch_size, n_lstm_steps, n_words, init_b=None,from_image=False,n_input=None,n_lstm_input=None,n_z=None):

        self.dim_in = dim_in
        self.dim_embed = dim_embed
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.n_lstm_steps = n_lstm_steps
        self.n_words = n_words
        self.n_input = n_input
        self.n_lstm_input=n_lstm_input
        self.n_z=n_z
        
        if from_image: 
            with open(vgg_path,'rb') as f:
                fileContent = f.read()
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(fileContent)
            self.images = tf.placeholder("float32", [1, 224, 224, 3])
            tf.import_graph_def(graph_def, input_map={"images":self.images})
            graph = tf.get_default_graph()
            self.sess = tf.InteractiveSession(graph=graph)

        self.from_image=from_image

        # declare the variables to be used for our word embeddings
        self.word_embedding = tf.Variable(tf.random_uniform([self.n_z, self.dim_embed], -0.1, 0.1), name='word_embedding')

        self.embedding_bias = tf.Variable(tf.zeros([dim_embed]), name='embedding_bias')
        
        # declare the LSTM itself
        self.lstm = tf.contrib.rnn.BasicLSTMCell(dim_hidden)
        self.dlstm = tf.contrib.rnn.BasicLSTMCell(n_lstm_input)
        
        # declare the variables to be used to embed the image feature embedding to the word embedding space
        self.img_embedding = tf.Variable(tf.random_uniform([dim_in, dim_hidden], -0.1, 0.1), name='img_embedding')
        self.img_embedding_bias = tf.Variable(tf.zeros([dim_hidden]), name='img_embedding_bias')

        # declare the variables to go from an LSTM output to a word encoding output
        self.word_encoding = tf.Variable(tf.random_uniform([dim_hidden, self.n_input], -0.1, 0.1), name='word_encoding')
        # initialize this bias variable from the preProBuildWordVocab output
        # optional initialization setter for encoding bias variable 
        if init_b is not None:
            self.word_encoding_bias = tf.Variable(init_b, name='word_encoding_bias')
        else:
            self.word_encoding_bias = tf.Variable(tf.zeros([self.n_input]), name='word_encoding_bias')
        with tf.device('/cpu:0'):
            self.embw=tf.Variable(xavier_init(self.n_input,self.n_z),name='embw')
        self.embb=tf.Variable(tf.zeros([self.n_z]),name='embb')
        self.all_encoding_weights=[self.embw,self.embb]
        self.auxy_in=tf.placeholder(tf.int32,[self.batch_size,2,self.n_lstm_steps])
        self.auxy=tf.reshape(self.auxy_in,[self.batch_size*2,-1])
        self.Xaux_in=tf.placeholder(tf.int32,[self.batch_size,2,1])
        self.Xaux=tf.reshape(self.Xaux_in,[self.batch_size*2])
        self.auxmask_in=tf.placeholder(tf.float32,[self.batch_size,2,self.n_lstm_steps])
        self.auxmask=tf.reshape(self.auxmask_in,[self.batch_size*2,-1])
        self.auxchoices_in=tf.placeholder(tf.float32,[self.batch_size,2,1])
        self.auxchoices=tf.reshape(self.auxchoices_in,[self.batch_size*2,-1])
        self.flatauxchoices=tf.reshape(self.auxchoices,[-1])


    def build_model(self):
        # declaring the placeholders for our extracted image feature vectors, our caption, and our mask
        # (describes how long our caption is with an array of 0/1 values of length `maxlen`  
        img = tf.placeholder(tf.float32, [self.batch_size, self.dim_in])
        caption_placeholder = tf.placeholder(tf.int32, [self.batch_size, self.n_lstm_steps])
        mask = tf.placeholder(tf.float32, [self.batch_size, self.n_lstm_steps])
        self.output_placeholder = tf.placeholder(tf.int32, [self.batch_size, self.n_lstm_steps])

        network_weights = self._initialize_weights()
        self.network_weights=network_weights
        # getting an initial LSTM embedding from our image_imbedding
        image_embedding = tf.matmul(img, self.img_embedding) + self.img_embedding_bias
        
        flat_caption_placeholder=tf.reshape(caption_placeholder,[-1])

        #leverage one-hot sparsity to lookup embeddings fast
        embedded_input,KLD_loss=self._get_word_embedding([network_weights['variational_encoding'],network_weights['biases_variational_encoding']],network_weights['input_meaning'],flat_caption_placeholder,logit=True,ret_z=False)
        # embedded_input=tf.stop_gradient(embedded_input)

        KLD_loss=tf.multiply(KLD_loss,tf.reshape(mask,[-1,1]))
        KLD_loss=tf.reduce_sum(KLD_loss)
        # KLD_loss=tf.stop_gradient(KLD_loss)
        # with tf.device('/cpu:0'):
        #     word_embeddings=tf.nn.embedding_lookup(self.embw,flat_caption_placeholder)
        # word_embeddings+=self.embb
        word_embeddings=tf.reshape(embedded_input,[self.batch_size,self.n_lstm_steps,-1])
        embedded_input=tf.reshape(embedded_input,[self.batch_size,self.n_lstm_steps,-1])
        # embedded_input=tf.nn.l2_normalize(embedded_input,dim=-1)
        #initialize lstm state
        state = self.lstm.zero_state(self.batch_size, dtype=tf.float32)
        rnn_output=[]
        total_loss=0
        self.deb=[]
        with tf.variable_scope("RNNcapt"):
            # unroll lstm
            for i in range(self.n_lstm_steps): 
                if i > 0:
                   # if this isn’t the first iteration of our LSTM we need to get the word_embedding corresponding
                   # to the (i-1)th word in our caption 
                    
                    current_embedding = word_embeddings[:,i-1,:]
                else:
                     #if this is the first iteration of our LSTM we utilize the embedded image as our input 
                    current_embedding = image_embedding
                if i > 0: 
                    # allows us to reuse the LSTM tensor variable on each iteration
                    tf.get_variable_scope().reuse_variables()

                out, state = self.lstm(current_embedding, state)
                if i>0:
                    logit=tf.matmul(out,self.word_encoding)+self.word_encoding_bias
                    # total_loss+=tf.reduce_sum(tf.reduce_sum(tf.square((tf.matmul(out,self.word_encoding)+self.word_encoding_bias)-embedded_input[:,i,:]),axis=-1)*mask[:,i])
                    # normed_embedding= tf.nn.l2_normalize(out, dim=-1)
                    # normed_target=tf.nn.l2_normalize(embedded_input[:,i,:],dim=-1)
                    # cos_sim=tf.multiply(normed_embedding,normed_target)
                    # cos_sim=(tf.reduce_sum(cos_sim,axis=-1))
                    # # cos_sim=tf.reshape(cos_sim,[self.batch_size,-1])
                    # cos_sim=tf.reduce_sum(cos_sim*mask[:,i])
                    # total_loss+=cos_sim
                    print logit.shape
                    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=caption_placeholder[:,i])

                    
                    xentropy = xentropy * mask[:,i]
                    xentropy=tf.reduce_sum(xentropy)
                    
                    total_loss+=xentropy
        #perform classification of output
        # rnn_output=tf.concat(rnn_output,axis=1)
        # rnn_output=tf.reshape(rnn_output,[self.batch_size*(self.n_lstm_steps),-1])
        # encoded_output=tf.matmul(rnn_output,self.word_encoding)+self.word_encoding_bias
        # encoded_output=tf
        # #get loss

        # # normed_embedding= tf.nn.l2_normalize(encoded_output, dim=-1)
        # # normed_target=tf.nn.l2_normalize(embedded_input,dim=-1)
        # # cos_sim=tf.multiply(normed_embedding,normed_target)[:,1:]
        # # cos_sim=(tf.reduce_sum(cos_sim,axis=-1))
        # # cos_sim=tf.reshape(cos_sim,[self.batch_size,-1])
        # # cos_sim=tf.reduce_sum(cos_sim[:,1:]*mask[:,1:])
        # # cos_sim=cos_sim/tf.reduce_sum(mask[:,1:])
        # # self.exp_loss=tf.reduce_sum((-cos_sim))
        # # # self.exp_loss=tf.reduce_sum(xentropy)/float(self.batch_size)
        # # total_loss = tf.reduce_sum(-(cos_sim))
        # mse=tf.reduce_sum(tf.reshape(tf.square(encoded_output-embedded_input),[self.batch_size,self.n_lstm_steps,-1]),axis=-1)[:,1:]*(mask[:,1:])
        # mse=tf.reduce_sum(mse)/tf.reduce_sum(mask[:,1:])

        #average over timeseries length

        # total_loss=tf.reduce_sum(masked_xentropy)/tf.reduce_sum(mask[:,1:])
        # total_loss=mse
        self.print_loss=total_loss
        total_loss+=KLD_loss
        total_loss/=tf.reduce_sum(mask[:,1:])
        self.print_loss=total_loss
        total_loss+=self.get_aux_loss()
        return total_loss, img,  caption_placeholder, mask
    def get_aux_loss(self):
        start_token_tensor=tf.constant((np.zeros([self.batch_size,binary_dim])).astype(np.float32),dtype=tf.float32)
        network_weights=self.network_weights
        seqlen=tf.cast(tf.reduce_sum(self.auxmask,reduction_indices=-1),tf.int32)
        

        

        KLD_penalty=1e-3

        # Use recognition network to determine mean and 
        # (log) variance of Gaussian distribution in latent
        # space
        if not same_embedding:
            input_embedding,input_embedding_KLD_loss=self._get_input_embedding([network_weights['variational_encoding'],network_weights['biases_variational_encoding']],network_weights['input_meaning'])
        else:
            input_embedding,input_embedding_KLD_loss=self._get_input_embedding([network_weights['variational_encoding'],network_weights['biases_variational_encoding']],network_weights['LSTM'])


        state = self.dlstm.zero_state(self.batch_size*2, dtype=tf.float32)

        loss = 0
        self.debug=0
        probs=[]
        with tf.variable_scope("RNN"):
            for i in range(self.n_lstm_steps): 
                if i > 0:

                    # current_embedding = tf.nn.embedding_lookup(self.word_embedding, caption_placeholder[:,i-1]) + self.embedding_bias    
                    current_embedding,KLD_loss = self._get_word_embedding([network_weights['variational_encoding'],network_weights['biases_variational_encoding']],network_weights['LSTM'], self.auxy[:,i-1])
                    current_embedding=tf.matmul(current_embedding,network_weights['LSTM']['affine_weight'])+network_weights['LSTM']['affine_bias']
                    # if transfertype2:
                    #     current_embedding=tf.stop_gradient(current_embedding)
                    loss+=tf.reduce_sum(KLD_loss*self.auxmask[:,i]*self.flatauxchoices)*KLD_penalty
                else:
                     current_embedding = input_embedding
                if i > 0: 
                    tf.get_variable_scope().reuse_variables()

                out, state = self.dlstm(current_embedding, state)

                
                if i > 0: 
                    
                    onehot=self.auxy[:,i]

                    logit = tf.matmul(out, network_weights['LSTM']['encoding_weight']) + network_weights['LSTM']['encoding_bias']
                    # if not use_ctc:
                    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=onehot)

                    
                    xentropy = xentropy * self.auxmask[:,i]*self.flatauxchoices
                    xentropy=tf.reduce_sum(xentropy)
                    # self.debug+=xentropy
                    loss += xentropy
                    self.deb.append(xentropy)
                    # else:
                    #     probs.append(tf.expand_dims(tf.nn.sigmoid(logit),1))
            # if not use_ctc:
                # loss_ctc=0
                # self.debug=self.debug/tf.reduce_sum(self.mask[:,1:])
            # else:
            #     probs=tf.concat(probs,axis=1)

            #     probs=ctc_loss.get_output_probabilities(probs,self.auxy[:,1:,:])
            #     loss_ctc=ctc_loss.loss(probs,self.auxy[:,1:,:],self.n_lstm_steps-2,self.batch_size,seqlen-1)
                
            # self.debug=tf.reduce_sum(input_embedding_KLD_loss)/self.batch_size*KLD_penalty+loss_ctc
            self.aux_loss = (loss / tf.reduce_sum(self.auxmask[:,1:]*self.auxchoices))
            self.aux_KLD=tf.reduce_sum(input_embedding_KLD_loss*self.flatauxchoices)*KLD_penalty#+loss_ctc

            return self.aux_loss+self.aux_KLD

    def build_generator(self, maxlen, batchsize=1,from_image=False):
        #same setup as `build_model` function

        img = tf.placeholder(tf.float32, [self.batch_size, self.dim_in])
        image_embedding = tf.matmul(img, self.img_embedding) + self.img_embedding_bias
        state = self.lstm.zero_state(batchsize,dtype=tf.float32)

        #declare list to hold the words of our generated captions
        all_words = []
        with tf.variable_scope("RNN"):
            # in the first iteration we have no previous word, so we directly pass in the image embedding
            # and set the `previous_word` to the embedding of the start token ([0]) for the future iterations
            output, state = self.lstm(image_embedding, state)
            previous_word = tf.nn.embedding_lookup(self.word_embedding, [0]) + self.embedding_bias

            for i in range(maxlen):
                tf.get_variable_scope().reuse_variables()

                out, state = self.lstm(previous_word, state)


                # get a get maximum probability word and it's encoding from the output of the LSTM
                logit = tf.matmul(out, self.word_encoding) + self.word_encoding_bias
                best_word = tf.argmax(logit, 1)

                with tf.device("/cpu:0"):
                    # get the embedding of the best_word to use as input to the next iteration of our LSTM 
                    previous_word = tf.nn.embedding_lookup(self.word_embedding, best_word)

                previous_word += self.embedding_bias

                all_words.append(best_word)
        self.img=img
        self.all_words=all_words
        return img, all_words
    def _initialize_weights(self):
        all_weights = dict()
        trainability=True
        if not same_embedding:
            all_weights['input_meaning'] = {
                'affine_weight': tf.Variable(xavier_init(self.n_z, self.n_lstm_input),name='affine_weight',trainable=trainability),
                'affine_bias': tf.Variable(tf.zeros(self.n_lstm_input),name='affine_bias',trainable=trainability)}
        with tf.device('/cpu:0'):
            om=tf.Variable(xavier_init(self.n_input, self.n_z),name='out_mean',trainable=trainability)
        all_weights['biases_variational_encoding'] = {
            'out_mean': tf.Variable(tf.zeros([self.n_z], dtype=tf.float32),name='out_meanb',trainable=trainability),
            'out_log_sigma': tf.Variable(tf.zeros([self.n_z], dtype=tf.float32),name='out_log_sigmab',trainable=trainability)}
        all_weights['variational_encoding'] = {
            'out_mean': om,
            'out_log_sigma': tf.Variable(xavier_init(self.n_input, self.n_z),name='out_log_sigma',trainable=trainability)}            
        # self.no_reload+=all_weights['input_meaning'].values()
        # self.var_embs=[]
        # if transfertype2:
        #     self.var_embs=all_weights['biases_variational_encoding'].values()+all_weights['variational_encoding'].values()

        # self.lstm=tf.contrib.rnn.BasicLSTMCell(n_lstm_input)
        # if lstm_stack>1:
        #     self.lstm=tf.contrib.rnn.MultiRNNCell([self.lstm]*lstm_stack)
        all_weights['LSTM'] = {
            'affine_weight': tf.Variable(xavier_init(self.n_z, self.n_lstm_input),name='affine_weight2'),
            'affine_bias': tf.Variable(tf.zeros(self.n_lstm_input),name='affine_bias2'),
            'encoding_weight': tf.Variable(xavier_init(self.n_lstm_input,self.n_input),name='encoding_weight'),
            'encoding_bias': tf.Variable(tf.zeros(self.n_input),name='encoding_bias')
            }
        all_encoding_weights=[all_weights[x].values() for x in all_weights]
        
        for w in all_encoding_weights:
            self.all_encoding_weights+=w
        all_weights['LSTM']['lstm']= self.dlstm
        return all_weights
    def _get_input_embedding(self, ve_weights, aff_weights):
        print self.Xaux.shape
        z,vae_loss=self._vae_sample(ve_weights[0],ve_weights[1],self.Xaux,lookup=True,sample=True)
        embedding=tf.matmul(z,aff_weights['affine_weight'])+aff_weights['affine_bias']

        return embedding,vae_loss
    def _get_word_embedding(self, ve_weights, lstm_weights, x,logit=False,ret_z=True):
        # x=tf.matmul(x,self.embw)+self.embb
        if logit:
            z,vae_loss=self._vae_sample(ve_weights[0],ve_weights[1],x,lookup=True)
        else:
            if not form2:
                z,vae_loss=self._vae_sample(ve_weights[0],ve_weights[1],x, True)
            else:
                z,vae_loss=self._vae_sample(ve_weights[0],ve_weights[1],tf.one_hot(x,depth=self.n_input))
                # all_the_f_one_h.append(tf.one_hot(x,depth=self.n_input))

        embedding=tf.matmul(z,self.word_embedding)+self.embedding_bias
        if ret_z:
            embedding=z
        return embedding,vae_loss
    def _vae_sample(self, weights, biases, x, lookup=False,sample=False):
            #TODO: consider adding a linear transform layer+relu or softplus here first 
            if not lookup:
                mu=tf.matmul(x,weights['out_mean'])+biases['out_mean']
                if not vanilla or sample:
                    logvar=tf.matmul(x,weights['out_log_sigma'])+biases['out_log_sigma']
            else:
                with tf.device('/cpu:0'):
                    mu=tf.nn.embedding_lookup(weights['out_mean'],x)
                mu+=biases['out_mean']
                if not vanilla or sample:
                    with tf.device('/cpu:0'):
                        logvar=tf.nn.embedding_lookup(weights['out_log_sigma'],x)
                    logvar+=biases['out_log_sigma']

            if not vanilla or sample:
                epsilon=tf.random_normal(tf.shape(logvar),name='epsilon')
                std=tf.exp(.5*logvar)
                z=mu+tf.multiply(std,epsilon)
            else:
                z=mu
            KLD=0.0
            if not vanilla or sample:
                KLD = -0.5 * tf.reduce_sum(1 + logvar - tf.pow(mu, 2) - tf.exp(logvar),axis=-1)
                print logvar.shape,epsilon.shape,std.shape,z.shape,KLD.shape
            return z,KLD
    def crop_image(self,x, target_height=227, target_width=227, as_float=True,from_path=True):
        #image preprocessing to crop and resize image
        image = (x)
        if from_path==True:
            image=cv2.imread(image)
        if as_float:
            image = image.astype(np.float32)

        if len(image.shape) == 2:
            image = np.tile(image[:,:,None], 3)
        elif len(image.shape) == 4:
            image = image[:,:,:,0]

        height, width, rgb = image.shape
        if width == height:
            resized_image = cv2.resize(image, (target_height,target_width))

        elif height < width:
            resized_image = cv2.resize(image, (int(width * float(target_height)/height), target_width))
            cropping_length = int((resized_image.shape[1] - target_height) / 2)
            resized_image = resized_image[:,cropping_length:resized_image.shape[1] - cropping_length]

        else:
            resized_image = cv2.resize(image, (target_height, int(height * float(target_width) / width)))
            cropping_length = int((resized_image.shape[0] - target_width) / 2)
            resized_image = resized_image[cropping_length:resized_image.shape[0] - cropping_length,:]

        return cv2.resize(resized_image, (target_height, target_width))

    def read_image(self,path=None):
        # parses image from file path and crops/resizes
        if path is None:
            path=test_image_path
        img = crop_image(path, target_height=224, target_width=224)
        if img.shape[2] == 4:
            img = img[:,:,:3]

        img = img[None, ...]
        return img

    def get_caption(self,x=None):
        #gets caption from an image by feeding it through imported VGG16 graph
        if self.from_image:
            feat = read_image(x)
            fc7 = self.sess.run(graph.get_tensor_by_name("import/Relu_1:0"), feed_dict={self.images:feat})
        else:
            fc7=np.load(x,'r')
        
        generated_word_index= self.sess.run(self.generated_words, feed_dict={self.img:fc7})
        generated_word_index = np.hstack(generated_word_index)
        generated_words = [ixtoword[x] for x in generated_word_index]
        punctuation = np.argmax(np.array(generated_words) == '.')+1

        generated_words = generated_words[:punctuation]
        generated_sentence = ' '.join(generated_words)
        return (generated_sentence)
def get_data(annotation_path, feature_path):
    #load training/validation data
    annotations = pd.read_table(annotation_path, sep='\t', header=None, names=['image', 'caption'])
    return np.load(feature_path,'r'), annotations['caption'].values
def preProBuildWordVocab(sentence_iterator, word_count_threshold=30): # function from Andre Karpathy's NeuralTalk
    #process and vectorize training/validation captions
    print('preprocessing %d word vocab' % (word_count_threshold, ))
    word_counts = {}
    nsents = 0
    for sent in sentence_iterator:
      nsents += 1
      for w in sent.lower().split(' '):
        word_counts[w] = word_counts.get(w, 0) + 1
    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print('preprocessed words %d -> %d' % (len(word_counts), len(vocab)))

    ixtoword = {}
    ixtoword[0] = '.'  
    wordtoix = {}
    wordtoix['#START#'] = 0 
    ix = 1
    for w in vocab:
      wordtoix[w] = ix
      ixtoword[ix] = w
      ix += 1

    word_counts['.'] = nsents
    bias_init_vector = np.array([1.0*word_counts[ixtoword[i]] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector) 
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) 
    return wordtoix, ixtoword, bias_init_vector.astype(np.float32)

dim_embed = 256
dim_hidden = 256
dim_in = 4096
batch_size = 5
momentum = 0.9
n_epochs = 3

def train(learning_rate=0.001, continue_training=False):
    
    tf.reset_default_graph()

    feats, captions = get_data(annotation_path, feature_path)
    wordtoix, ixtoword, init_b = preProBuildWordVocab(captions)

    np.save('data/ixtoword', ixtoword)

    print ('num words:',len(ixtoword))

    


    sess = tf.InteractiveSession()
    n_words = len(wordtoix)
    maxlen = 30
    X, final_captions, captmask, _map, auxy,auxmask,Xaux,auxchoices = load_text(50000-2,captions)
    running_decay=1
    decay_rate=0.9999302192204246
    # with tf.device('/gpu:0'):
    caption_generator = Caption_Generator(dim_in, dim_hidden, dim_embed, batch_size, maxlen+2, n_words, np.zeros(n_input).astype(np.float32),n_input=n_input,n_lstm_input=n_lstm_input,n_z=n_z)

    loss, image, sentence, mask = caption_generator.build_model()

    saver = tf.train.Saver(max_to_keep=100)
    print [x.name for x in tf.trainable_variables()]
    caption_generator.all_encoding_weights+=[x for x in tf.trainable_variables() if x.name.startswith('RNN/') ]
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    tf.global_variables_initializer().run()
    tf.train.Saver(var_list=caption_generator.all_encoding_weights,max_to_keep=100).restore(sess,tf.train.latest_checkpoint('modelsvarvaedefoh'))
    if continue_training:
        saver.restore(sess,tf.train.latest_checkpoint(model_path))
    losses=[[],[],[],[]]

    for epoch in range(n_epochs):
        if epoch==1:
            for w in caption_generator.all_encoding_weights:
                w.trainable=True
        index = (np.arange(len(feats)).astype(int))
        np.random.shuffle(index)
        index=index[:]
        i=0
        for start, end in zip( range(0, len(index), batch_size), range(batch_size, len(index), batch_size)):
            if i%1000==0 and i!=0:
                break
            #format data batch
            current_feats = feats[index[start:end]]
            current_captions = captions[index[start:end]]
            current_caption_ind = [x for x in map(lambda cap: [wordtoix[word] for word in cap.lower().split(' ')[:-1] if word in wordtoix], current_captions)]

            current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=maxlen+1)
            current_caption_matrix = np.hstack( [np.full( (len(current_caption_matrix),1), 0), current_caption_matrix] )

            current_mask_matrix = np.zeros((current_caption_matrix.shape[0], current_caption_matrix.shape[1]))
            nonzeros = np.array([x for x in map(lambda x: (x != 0).sum()+2, current_caption_matrix )])
            current_capts=final_captions[index[start:end]]
            

            for ind, row in enumerate(current_mask_matrix):
                row[:nonzeros[ind]] = 1
            current_mask_matrix=captmask[index[start:end]]

            _, loss_value,total_loss,aux_KLD,aux_loss = sess.run([train_op, caption_generator.print_loss,loss,caption_generator.aux_KLD,caption_generator.aux_loss], feed_dict={
                image: current_feats.astype(np.float32),
                caption_generator.output_placeholder : current_caption_matrix.astype(np.int32),
                mask : current_mask_matrix.astype(np.float32),
                sentence : current_capts.astype(np.float32),
                caption_generator.auxy_in:auxy[index[start:end]][:,:2],
                caption_generator.Xaux_in:Xaux[index[start:end]][:,:2],
                caption_generator.auxmask_in:auxmask[index[start:end]][:,:2],
                caption_generator.auxchoices_in:auxchoices[index[start:end]][:,:2]
                })

            print("Current Cost: ", loss_value, "\t Epoch {}/{}".format(epoch, n_epochs), "\t Iter {}/{}".format(start,len(feats)))
            losses[0].append(loss_value)
            losses[1].append(aux_loss)
            losses[2].append(aux_KLD)
            losses[3].append(total_loss)
            # losses.append(loss_value*running_decay)
            # if epoch<9:
            #     if i%3==0:
            #         running_decay*=decay_rate
            # else:
            #     if i%8==0:
            #         running_decay*=decay_rate
            i+=1
            print [x[-1] for x in losses]
            # print deb
        print("Saving the model from epoch: ", epoch)
        pkl.dump(losses,open('losses/loss_e2e_aux_init.pkl','wb'))
        saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)
        learning_rate *= 0.95
def test(sess,image,generated_words,ixtoword,idx=0): # Naive greedy search

    feats, captions = get_data(annotation_path, feature_path)
    feat = np.array([feats[idx]])
    
    saver = tf.train.Saver()
    sanity_check= False
    # sanity_check=True
    if not sanity_check:
        saved_path=tf.train.latest_checkpoint(model_path)
        saver.restore(sess, saved_path)
    else:
        tf.global_variables_initializer().run()

    generated_word_index= sess.run(generated_words, feed_dict={image:feat})
    generated_word_index = np.hstack(generated_word_index)

    generated_sentence = [ixtoword[x] for x in generated_word_index]
    print(generated_sentence)

if __name__=='__main__':

    model_path = './models/tensorflow_aux_init'
    feature_path = './data/feats.npy'
    annotation_path = './data/results_20130124.token'
    import sys
    feats, captions = get_data(annotation_path, feature_path)
    n_input=50000
    binary_dim=n_input
    n_lstm_input=256
    n_z=500
    zero_end_tok=True
    form2=True
    vanilla=True
    onehot=False
    same_embedding=False

    if sys.argv[1]=='train':
        train()
    elif sys.argv[1]=='test':
        ixtoword = np.load('data/ixtoword.npy').tolist()
        n_words = len(ixtoword)
        maxlen=15
        sess = tf.InteractiveSession()
        batch_size=1
        caption_generator = Caption_Generator(dim_in, dim_hidden, dim_embed, 1, maxlen+2, n_words,n_input=n_input,n_lstm_input=n_lstm_input,n_z=n_z)


        image, generated_words = caption_generator.build_generator(maxlen=maxlen)
        test(sess,image,generated_words,ixtoword,1)