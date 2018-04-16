# Copyright 2018 Sanjay Dorairaj. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf
import numpy as np
import pickle
import os
import time
import shutil

import utils
import batch_helper

def display_stats(params,train_batch,test_batch,epoch_index,batches_count,loss_value,train_preds,test_preds):
    
    # extract data from train and test batches for display
    (raw_enc_in_batch,enc_in_batch,enc_in_batch_len,
        raw_dec_in_batch,dec_in_batch,dec_in_batch_len,
        raw_dec_out_batch,dec_out_batch,dec_out_batch_len) = train_batch
        
    (test_raw_enc_in_batch,test_enc_in_batch,test_enc_in_batch_len,
        test_raw_dec_in_batch,test_dec_in_batch,test_dec_in_batch_len,
        test_raw_dec_out_batch,test_dec_out_batch,test_dec_out_batch_len) = test_batch
    
    # length of output story and summary                    
    sample_id = np.random.randint(0,params.batch_size)

    # displaying training results
    print("Epoch:%d,Completed Batches:%d, Loss:%0.4f" % (epoch_index,batches_count,loss_value))
    print("Train. Story       :"," ".join(raw_enc_in_batch[sample_id][:params.max_display_len]))
    print("Train. Orig Summary:", " ".join(raw_dec_in_batch[sample_id][:params.max_display_len]))

    #print("type(train_preds):%s,train_preds.shape:%s" % (type(train_preds),train_preds.shape))

    train_preds = np.transpose(train_preds)
    train_pred_ids = train_preds[sample_id]

    #print("preds:\n",train_preds)
    train_summary = [params.summary_dicts[1][x] for x in train_pred_ids]

    print("Train. New  Summary:"," ".join(train_summary[:
                        params.max_display_len if len(train_summary)<params.max_display_len else len(train_summary)])) 

    # display testing results       
    if len(test_raw_enc_in_batch) < sample_id:
        print("len(test_raw_enc_in_batch) < sample_id...resetting sample_id to 0")
        print("test_raw_enc_in_batch:\n",test_raw_enc_in_batch)
        sample_id = 0

    #print("len(test_raw_enc_in_batch[sample_id]):",len(test_raw_enc_in_batch[sample_id]))
    print("\nTest. Story       :"," ".join(test_raw_enc_in_batch[sample_id][:
                        params.max_display_len if len(test_raw_enc_in_batch[sample_id]) > params.max_display_len else len(test_raw_enc_in_batch[sample_id])]))
    print("Test. Original Summary:", " ".join(test_raw_dec_in_batch[sample_id][:params.max_display_len if len(test_raw_dec_in_batch[sample_id])>params.max_display_len else len(test_raw_dec_in_batch[sample_id])]))  

    #print("type(test_preds):%s,len(test_preds):%d" % (type(test_preds),len(test_preds)))
    #print("test_preds:\n",test_preds)

    test_preds = test_preds[0]

    #print("type(test_preds):%s,test_preds.shape:%s" % (type(test_preds),test_preds.shape))

    test_preds = np.transpose(test_preds)
    test_pred_ids = test_preds[sample_id]

    #print("test_preds:\n",test_preds)
    test_summary = [params.summary_dicts[1][x] for x in test_pred_ids]                                                    
    print("Test. New Summary:"," ".join(test_summary[:
                params.max_display_len if len(test_summary)>params.max_display_len else len(test_summary)]))  

    print("-"*80)   
    
    # dump stats for rouge computation
    dump_data_for_rouge_score(params,test_raw_dec_in_batch,test_preds)
    
def dump_data_for_rouge_score(params,test_raw_dec_in_batch,test_preds):
    
    if len(test_raw_dec_in_batch) != len(test_preds):
        print("Reference/Hypothesis count mismatch.len(test_raw_dec_in_batch):%d, len(test_preds):%d" % (len(test_raw_dec_in_batch),len(test_preds)))
        return
    
    # open files to write metrics
    hyp_file = open(params.rouge_evaluation_dir + "/hypothesis.txt", "w")
    ref_file = open(params.rouge_evaluation_dir + "/reference.txt", "w")    
    
    for i in range(len(test_raw_dec_in_batch)):
        
        reference = " ".join(test_raw_dec_in_batch[i][1:][:params.max_display_len if len(test_raw_dec_in_batch[i])>params.max_display_len else len(test_raw_dec_in_batch[i])])
        
        test_pred_ids = test_preds[i]
        test_summary = [params.summary_dicts[1][x] for x in test_pred_ids]  
        hypothesis = " ".join(test_summary[:
                params.max_display_len if len(test_summary)>params.max_display_len else len(test_summary)])
        
        hyp_file.write(hypothesis + "\n")
        ref_file.write(reference + "\n")
        
    hyp_file.close()
    ref_file.close()
        
def initialize_data(params):
    
    params.final_vocab_size = params.vocab_size + 4
    
    story_vocab_file_path = params.data_dir + "/" + params.story_vocab_file
    summary_vocab_file_path = params.data_dir + "/" + params.summary_vocab_file
    
    params.story_dicts = utils.getVocabDicts(params,story_vocab_file_path)
    params.summary_dicts = utils.getVocabDicts(params,summary_vocab_file_path)
    
    if params.regenerate_dataset:
        # recreate the train and test datasets
        print("Recreating train and test datasets")
        utils.create_train_and_test_datasets(params)
    else:
        # Restore dataset from pickle files
        print("Restoring train and test datasets")
        utils.restore_train_and_test_datasets(params)

def create_model(params):

    # create default graph
    tf.reset_default_graph()

    with tf.name_scope("Inputs"):

        # shape of input is [max_time,batch_size] - time_major
        params.encoder_inputs = tf.placeholder(tf.int32,[None,None],name="encoder_inputs")
        params.decoder_inputs = tf.placeholder(tf.int32,[None,None],name="decoder_inputs")
        params.decoder_targets = tf.placeholder(tf.int32,[None,None],name="decoder_targets")

        params.source_sequence_length = tf.placeholder(tf.int32,shape=[params.batch_size])
        params.target_sequence_length = tf.placeholder(tf.int32,shape=[params.batch_size])

        params.tf_batch_size = tf.size(params.source_sequence_length)
    
    # Embedding
    params.emb_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)

    # Encoder
    with tf.variable_scope('encoder_lstm_cell'):

        params.encoder_cell = tf.contrib.rnn.BasicLSTMCell(params.hidden_units,name="encoder_lstm_cell")
        params.encoder_cell = tf.nn.rnn_cell.DropoutWrapper(params.encoder_cell,
                                                        output_keep_prob=params.keep_prob)

        params.encoder_emb = tf.get_variable("embedding_encoder",[params.final_vocab_size, params.embedding_size], initializer=params.emb_init)

        params.encoder_emb_inp = tf.nn.embedding_lookup(params.encoder_emb, params.encoder_inputs)

        # initialize the initial hidden state
        params.initial_hidden_state = params.encoder_cell.zero_state(params.tf_batch_size,dtype=tf.float32)

        # Run Dynamic RNN
        # encoder_outputs: [max_time,batch_size,hidden_units]
        # encoder_state: [batch_size,hidden_units]
        # sequence length is a vector of length batch_size
        params.encoder_outputs, params.encoder_state = tf.nn.dynamic_rnn(
                                                        params.encoder_cell,
                                                        params.encoder_emb_inp,
                                                        initial_state = params.initial_hidden_state,
                                                        sequence_length=params.source_sequence_length,
                                                        time_major=True
                                                        )

    ## Decoder Embedding matrix
    with tf.variable_scope('decoder_embeddings'):
            params.decoder_emb = tf.get_variable(
                "embedding_decoder", [params.final_vocab_size, params.embedding_size],
                initializer=params.emb_init)

    # Build RNN Cell
    with tf.variable_scope('train_decoder'):

        # embedding lookup
        params.decoder_emb_inp = tf.nn.embedding_lookup(params.decoder_emb, params.decoder_inputs)
        
        params.train_attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                  num_units=params.hidden_units, 
                  #memory = tf.reshape(params.encoder_outputs,[params.batch_size,params.encoder_max_time,params.hidden_units]),
                  memory = tf.transpose(params.encoder_outputs, [1, 0, 2]),
                  memory_sequence_length=params.source_sequence_length,
                  normalize = True
        )

        # basic LSTM decoder cell
        params.train_decoder_cell = tf.contrib.rnn.BasicLSTMCell(params.hidden_units,name="train_decoder_lstm_cell")
        
        params.train_attn_cell = tf.contrib.seq2seq.AttentionWrapper(
                params.train_decoder_cell , params.train_attention_mechanism, 
                    alignment_history=True, attention_layer_size=params.hidden_units)

        # add projection layer
        params.train_decoder_cell = tf.contrib.rnn.OutputProjectionWrapper(params.train_attn_cell,params.final_vocab_size)

        # Helper
        params.train_helper = tf.contrib.seq2seq.TrainingHelper(
                                        params.decoder_emb_inp,
                                        params.target_sequence_length, 
                                        time_major=True)

        # Decoder
        initial_state = params.train_decoder_cell.zero_state(params.batch_size, params.dtype).clone(
                                                                          cell_state=params.encoder_state)
        params.train_decoder = tf.contrib.seq2seq.BasicDecoder(
                                cell = params.train_decoder_cell, 
                                helper = params.train_helper, 
                                #initial_state=params.train_decoder_cell.zero_state(
                                #    dtype=tf.float32, batch_size=params.batch_size))
                                initial_state = initial_state)

        # Dynamic decoding
        params.train_output_states,params.train_final_context_state,_ = tf.contrib.seq2seq.dynamic_decode(
                                                                    params.train_decoder,
                                                                    output_time_major = True)
        # logits = [batch x seq_len x decoder_vocabulary_size]
        params.train_logits = params.train_output_states.rnn_output

        # predictions = [batch x seq_len]
        params.train_predictions = tf.identity(params.train_output_states.sample_id)

    ## Inference Decoder

    with tf.variable_scope('test_decoder'):
        
        # basic LSTM decoder cell
        params.test_decoder_cell = tf.contrib.rnn.BasicLSTMCell(params.hidden_units,name="test_decoder_lstm_cell")
        
        params.start_tokens = tf.tile(tf.constant([params.sentence_start_index],  
                           dtype=tf.int32),  
                           [params.batch_size], 
                           name='start_tokens')

        if params.inference_style == "greedy_search":

            params.test_attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                      num_units=params.hidden_units, 
                      memory = tf.transpose(params.encoder_outputs, [1, 0, 2]),
                      memory_sequence_length=params.source_sequence_length,
                      normalize = True
            )
    
            params.test_decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                    params.test_decoder_cell , params.test_attention_mechanism, 
                        alignment_history=True, attention_layer_size=params.hidden_units)

            # add projection layer
            params.test_decoder_cell = tf.contrib.rnn.OutputProjectionWrapper(params.test_decoder_cell,params.final_vocab_size)            
            params.test_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                                                      params.decoder_emb,
                                                      params.start_tokens,
                                                      params.sentence_end_index)

            # Decoder
            initial_state = params.test_decoder_cell.zero_state(params.batch_size, params.dtype).clone(
                                                                      cell_state=params.encoder_state)
            params.test_decoder = tf.contrib.seq2seq.BasicDecoder(
                                    cell = params.test_decoder_cell, 
                                    helper = params.test_helper, 
                                    #initial_state=params.test_decoder_cell.zero_state(
                                    #    dtype=tf.float32, batch_size=params.batch_size))
                                    initial_state = initial_state)

            # Decoder
            #params.test_decoder = tf.contrib.seq2seq.BasicDecoder(
            #                        cell = params.test_decoder_cell, 
            #                        helper = params.test_helper, 
            #                        initial_state = params.encoder_state)

            # Dynamic decoding
            params.test_output_states,params.test_final_context_state,_ = tf.contrib.seq2seq.dynamic_decode(
                                                                        params.test_decoder,
                                                                        maximum_iterations=params.max_summary_length,
                                                                        output_time_major = True)
            # logits = [batch x seq_len x decoder_vocabulary_size]
            params.test_logits = params.test_output_states.rnn_output

            # predictions = [batch x seq_len]
            params.test_predictions = tf.identity(params.test_output_states.sample_id)
            
        elif params.inference_style == "beam_search":
            
            # See https://www.tensorflow.org/tutorials/seq2seq for more information
            # and https://github.com/tensorflow/tensorflow/issues/11598
            # and https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/BeamSearchDecoder
            # and https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/seq2seq/python/kernel_tests/beam_search_decoder_test.py
            # and https://github.com/tensorflow/tensorflow/issues/13154 (Note that alignment history was disabled since it was not supported in BeamSearchDecoder
            
            initial_state = params.test_decoder_cell.zero_state(params.batch_size,params.dtype)
            
            #encoder_outputs_t = tf.transpose(params.encoder_outputs, [1, 0, 2])
            tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(
                                        params.encoder_outputs, multiplier=params.beam_width)
            
            tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(
                                        params.encoder_state, multiplier=params.beam_width)
            tiled_sequence_length = tf.contrib.seq2seq.tile_batch(
                                        params.source_sequence_length, multiplier=params.beam_width)
            
            params.test_attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                      num_units=params.hidden_units, 
                      memory = tiled_encoder_outputs,
                      memory_sequence_length=tiled_sequence_length,
                      normalize = True)            
            
            initial_state = tf.contrib.seq2seq.tile_batch(initial_state,multiplier=params.beam_width)    
            
            params.test_decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                                                params.test_decoder_cell , 
                                                params.test_attention_mechanism, 
                                                #alignment_history=True, 
                                                attention_layer_size=params.hidden_units)
            
            cell_state = params.test_decoder_cell.zero_state(params.batch_size * params.beam_width,params.dtype)
            cell_state = cell_state.clone(cell_state=initial_state)
            
            # add projection layer
            params.test_decoder_cell = tf.contrib.rnn.OutputProjectionWrapper(params.test_decoder_cell,params.final_vocab_size) 
            
            params.test_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell = params.test_decoder_cell,
                embedding = params.decoder_emb,
                start_tokens = params.start_tokens,
                end_token = params.sentence_end_index,
                initial_state = cell_state,
                beam_width = params.beam_width,
                length_penalty_weight = 0.0)
            
            # Dynamic decoding
            params.test_output_states,_,_ = tf.contrib.seq2seq.dynamic_decode(
                                                                        params.test_decoder,
                                                                        impute_finished = False,
                                                                        maximum_iterations=params.max_summary_length,
                                                                        output_time_major = True)
            
            #params.test_predictions = params.test_output_states.sample_id
            
            # get predictions
            translations = params.test_output_states
            params.test_predictions = tf.identity(translations[0])
            
            # convert to [batch_size,beam_width,time] format
            #translations = tf.transpose(translations, [1, 0, 2])
            
            # get the first prediction
            #params.test_predictions = tf.identity(translations[0])

    # Loss

    #_target_sequence_length = [100]*128
    params.weights = tf.sequence_mask(params.target_sequence_length, dtype=tf.float32)

    params.sequence_loss = tf.contrib.seq2seq.sequence_loss(
                                    params.train_logits,
                                    params.decoder_targets,
                                    params.weights,
                                    average_across_timesteps=False,
                                    average_across_batch=False)

    # I might have to divide by batch size to make the loss invariant to batch size 
    # See https://www.tensorflow.org/tutorials/seq2seq
    params.train_loss = tf.reduce_mean(params.sequence_loss)

    # Gradient computation and optimization
    # gradient computation
    tf_params = tf.trainable_variables()
    gradients = tf.gradients(params.train_loss,tf_params)
    clipped_gradients,_ = tf.clip_by_global_norm(gradients,params.max_grad_norm)

    # optimizer
    optimizer = tf.train.AdamOptimizer(params.learning_rate)

    params.update_step = optimizer.apply_gradients(zip(clipped_gradients,tf_params))

    
def train_loop(params):

    ## training and validation
    debug = False

    batch_stats_display_count = params.batch_stats_display_count
    batches_count = 0
    max_epochs = params.max_training_epochs
    total_batches = 0
    save_model = False
    params.train_batch_index = 0
    params.test_batch_index = 0

    if tf.gfile.Exists(params.logs_path):
       tf.gfile.DeleteRecursively(params.logs_path) 

    # saver for checkpointing model
    saver = tf.train.Saver(max_to_keep=4)

    with tf.Session() as sess:

        # initialize global variables
        sess.run(tf.global_variables_initializer())
            
        # restore/backup from/to existing checkpoints
        print("Checkpoint path:%s" % params.ckpt_path)
            
        # restore model if exists
        if params.ignore_checkpoint != True:
            try:
                saver.restore(sess,params.ckpt_path)
                print("Session restored from checkpoint:",params.ckpt_path)
            except Exception as exp: 
                print("Unable to restore from checkpoint. No checkpoint files found")
                pass
        else:
            print("ignore_checkpoint enabled. Not restoring from saved checkpoints")

        # backup any existing models
        backup_path = params.ckpt_dir + "_" + "-".join(time.ctime().split(" ")).replace(':','-')

        shutil.copytree(params.ckpt_dir,backup_path)
        print("Backed up previous model in:",backup_path)

        # Create a summary to monitor cost tensor
        tf.summary.scalar("train_loss", params.train_loss)

        # merge into a single op
        merged_summary_op = tf.summary.merge_all()

        # write logs to tensorboard
        summary_writer = tf.summary.FileWriter(params.logs_path, graph=tf.get_default_graph())

        # training loop
        for epoch_index in range(1,max_epochs+1):

            params.train_batch_index = 0
            batches_count = 0

            # Train
            while True:

                train_batch = batch_helper.getNextBatch(params,params.train_in_dataset,params.train_out_dataset)

                if train_batch == None:
                    break

                (raw_enc_in_batch,enc_in_batch,enc_in_batch_len,
                 raw_dec_in_batch,dec_in_batch,dec_in_batch_len,
                 raw_dec_out_batch,dec_out_batch,dec_out_batch_len) = train_batch

                # feed inputs
                feed_dict_train = {
                    params.encoder_inputs: enc_in_batch,
                    params.decoder_inputs: dec_in_batch,
                    params.decoder_targets:dec_out_batch,
                    params.source_sequence_length: enc_in_batch_len,
                    params.target_sequence_length: dec_out_batch_len
                }

                # training
                _, loss_value, summary, train_preds = sess.run([params.update_step, params.train_loss, merged_summary_op,params.train_predictions], feed_dict=feed_dict_train)
                
                # focus on just the first batch for now
                params.test_batch_index = 0
                test_batch = batch_helper.getNextBatch(params,params.test_in_dataset,params.test_out_dataset,training=False)

                (test_raw_enc_in_batch,test_enc_in_batch,test_enc_in_batch_len,
                 test_raw_dec_in_batch,test_dec_in_batch,test_dec_in_batch_len,
                 test_raw_dec_out_batch,test_dec_out_batch,test_dec_out_batch_len) = test_batch
                
                #print("Completed training...proceeding to test...")
                # feed inputs
                feed_dict_test = {
                    params.encoder_inputs: test_enc_in_batch,
                    params.source_sequence_length: test_enc_in_batch_len
                }
                
                # testing
                test_preds = sess.run([params.test_predictions], feed_dict=feed_dict_test)

                # increment batches_count and see if we need to display stats
                batches_count += 1    
                total_batches += 1
                
                # display stats and create checkpoint
                if batches_count % params.batch_stats_display_count == 0:
                    display_stats(params,train_batch,test_batch,epoch_index,batches_count,loss_value,train_preds,test_preds)
                    
                    # Create checkpoint
                    if params.save_model == True:
                        print("Creating checkpoint in:",params.ckpt_path)
                        saver.save(sess, params.ckpt_path)

                # update tensorboard summary
                summary_writer.add_summary(summary,total_batches)
                
def train(params):
    
    # load dataset 
    initialize_data(params)
    
    # create the tensorflow graph
    create_model(params)
    
    # run the training loop with intermediate validation
    train_loop(params)
    