#!/usr/bin/env python

##########################################
#
# Seq2seq.py: An implementation of seq2seq model using Tensorflow.
#
# Author: Cosimo Iaia <cosimo.iaia@gmail.com>
# Date: 05/05/2016
#
# This file is distribuited under the terms of GNU General Public
#
#########################################

import tensorflow as tf
import json
import numpy as np
import os, sys, re
import util.s2s_reader as s2s_reader


data_path = "data"
model_path = "output"

expression = r"[0-9]+|[']*[\w]+"

batch_size = 128


#data params
bucket_option = [i for i in range(1, 200, 5)]
buckets = s2s_reader.create_bucket(bucket_option)

reader = s2s_reader.reader(file_name = data_path, batch_size = batch_size, buckets = buckets, bucket_option = bucket_option, clean_mode=True)
vocab_size = len(reader.dict)

hidden_size = 512
projection_size = 300
embedding_size = 300
num_layers = 3

# ouput_size for softmax layer
output_size = projection_size

#training params, truncated_norm will resample x > 2std; so when std = 0.1, the range of x is [-0.2, 0.2]
truncated_std = 0.1
keep_prob = 0.95
max_epoch = 250
norm_clip = 5

#training params for adam
adam_learning_rate = 0.001


#model name & save path
model_name = "p"+str(projection_size)+"_h"+str(hidden_size)+"_x"+str(num_layers)
save_path = model_path+"/"+model_name


###### MODEL DEFINITION

tf.reset_default_graph()
sess = tf.InteractiveSession()

#placeholder
enc_inputs = tf.placeholder(tf.int32, shape=(None, batch_size), name="enc_inputs")
targets = tf.placeholder(tf.int32, shape=(None, batch_size), name="targets")
dec_inputs = tf.placeholder(tf.int32, shape=(None, batch_size), name="dec_inputs")

#input embedding layers
emb_weights = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], stddev=truncated_std), name="emb_weights")
enc_inputs_emb = tf.nn.embedding_lookup(emb_weights, enc_inputs, name="enc_inputs_emb")
dec_inputs_emb = tf.nn.embedding_lookup(emb_weights, dec_inputs, name="dec_inputs_emb")

#cell definiton
enc_cell_list=[]
dec_cell_list=[]

for i in range(num_layers):

	single_cell = tf.nn.rnn_cell.LSTMCell(
		num_units=hidden_size, 
		num_proj=projection_size, 
		#initializer=tf.truncated_normal_initializer(stddev=truncated_std),
		state_is_tuple=True
		)
	if i < num_layers-1 or num_layers == 1:
		single_cell = tf.nn.rnn_cell.DropoutWrapper(cell=single_cell, output_keep_prob=keep_prob)
	enc_cell_list.append(single_cell)

for i in range(num_layers):

	single_cell = tf.nn.rnn_cell.LSTMCell(
		num_units=hidden_size, 
		num_proj=projection_size, 
		#initializer=tf.truncated_normal_initializer(stddev=truncated_std),
		state_is_tuple=True
		)
	if i < num_layers-1 or num_layers == 1:
		single_cell = tf.nn.rnn_cell.DropoutWrapper(cell=single_cell, output_keep_prob=keep_prob)
	dec_cell_list.append(single_cell)



enc_cell = tf.nn.rnn_cell.MultiRNNCell(cells=enc_cell_list, state_is_tuple=True)
dec_cell = tf.nn.rnn_cell.MultiRNNCell(cells=dec_cell_list, state_is_tuple=True)

#encoder & decoder defintion
_, enc_states = tf.nn.dynamic_rnn(cell = enc_cell, 
	inputs = enc_inputs_emb, 
	dtype = tf.float32, 
	time_major = True, 
	scope="encoder")

dec_outputs, dec_states = tf.nn.dynamic_rnn(cell = dec_cell, 
	inputs = dec_inputs_emb, 
	initial_state = enc_states, 
	dtype = tf.float32, 
	time_major = True, 
	scope="decoder")

#output layers
project_w = tf.Variable(tf.truncated_normal(shape=[output_size, embedding_size], stddev=truncated_std), name="project_w")
project_b = tf.Variable(tf.constant(shape=[embedding_size], value = 0.1), name="project_b")
softmax_w = tf.Variable(tf.truncated_normal(shape=[embedding_size, vocab_size], stddev=truncated_std), name="softmax_w")
softmax_b = tf.Variable(tf.constant(shape=[vocab_size], value = 0.1), name="softmax_b")

dec_outputs = tf.reshape(dec_outputs, [-1, output_size], name="dec_ouputs")
dec_proj = tf.matmul(dec_outputs, project_w) + project_b
logits = tf.nn.log_softmax(tf.matmul(dec_proj, softmax_w) + softmax_b, name="logits")

#loss function
flat_targets = tf.reshape(targets, [-1])
total_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=flat_targets)
avg_loss = tf.reduce_mean(total_loss)

#optimization
optimizer = tf.train.AdamOptimizer(adam_learning_rate)
gvs = optimizer.compute_gradients(avg_loss)
capped_gvs = [(tf.clip_by_norm(grad, norm_clip), var) for grad, var in gvs]
train_op = optimizer.apply_gradients(capped_gvs)

#prediction
logit = logits[-1]
top_values, top_indexs = tf.nn.top_k(logit, k = 10, sorted=True)


#initialization or load model
saver = tf.train.Saver()


os.makedirs(save_path)
sess.run(tf.global_variables_initializer())
losses = []



#-----------------SUPPORT FUNCTIONS--------------

def update_summary(save_path, losses):
	summary_location = save_path + "/summary.json"
	if os.path.exists(summary_location):
		os.remove(summary_location)
	with open(summary_location, 'w') as outfile:
		json.dump(losses, outfile)




def translate(token_list):
	enc = []
	for token in token_list:
		if token in reader.dict:
			enc.append(reader.dict[token])
		else:
			enc.append(reader.dict['[unk]'])
	#dec will be append with 2 inside the model
	print(enc)
	return enc


# Let's roll:

# Make a nice progress bar during training
from tqdm import tqdm


#local variables
count = 0
epoch_loss = 0
epoch_count = 0

print("Training Bot...")
pbar=tqdm(total=max_epoch)
pbar.set_description("Epoch 1: Loss: 0 Avg loss: 0 Count: 0")


while True:

	curr_epoch = reader.epoch
	data,index = reader.next_batch()
	
	enc_inp, dec_inp, dec_tar = s2s_reader.data_processing(data, buckets[index], batch_size)

	if reader.epoch != curr_epoch:
		

		losses.append(epoch_loss/epoch_count)

		epoch_loss = 0
		epoch_count = 0

		update_summary(save_path, losses)
		cwd = os.getcwd()
		saver.save(sess, save_path+"/model.ckpt")

		if reader.epoch == (max_epoch+1):
			break

	feed_dict = {enc_inputs: enc_inp, dec_inputs:dec_inp, targets: dec_tar}
	_, loss_t = sess.run([train_op, avg_loss], feed_dict)
	epoch_loss += loss_t

	count+=1
	epoch_count+=1
	pbar.update(1)

	if count%10 == 0:
		pbar.set_description("Epoch "+str(reader.epoch) +": Loss: " + str(loss_t) + " Avg loss: " +str(epoch_loss/epoch_count) +" Count: "+ str(epoch_count * batch_size))

print("Training Complete!")