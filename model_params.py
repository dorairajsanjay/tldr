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

class ModelParams(object):
    batch_size = 0
    hidden_units = 0
    embedding_size = 0
    max_grad_norm = 0
    learning_rate = 0
    model_dir = ""
    ignore_checkpoint = False
    data_dir = ""
    generate_dataset = False
    regenerate_dataset = False
    story_vocab_file = ""
    summary_vocab_file = ""
    train_in_file = ""
    train_out_file = ""
    test_in_file = ""
    test_out_file = ""
    dev_in_file = "dev.in"
    dev_out_file = "dev.out"
    
    dtype = tf.float32
    
    save_model = True
    
    # special vocab tokens
    pad_token      = "<pad>"
    unknown_token  = "<unk>"
    sentence_start = "<sos>"
    sentence_end   = "<eos>"
    
    pad_token_index      = 0
    unknown_token_index  = 1
    sentence_start_index = 2
    sentence_end_index   = 3
    
    # indexes to keep track of batch processing
    train_batch_index = 0
    test_batch_index = 0
    
    # encoder and decoder lengths
    encoder_max_time = 300
    decoder_max_time = 100   
    
    # dicts for converting back and forth between vocab and index
    story_dicts   = None
    summary_dicts = None
    
    train_in_dataset  = None
    train_out_dataset = None
    test_in_dataset   = None
    test_out_dataset  = None
    dev_in_dataset    = None
    dev_out_dataset   = None
    
    final_vocab_size = None
    
    # max summary to generate
    max_summary_length = 20
    max_display_len = 12
    
    # for dropout
    keep_prob = 0.8

    max_training_epochs = 100
    batch_stats_display_count = 10
    
    # inference
    inference_style = "greedy_search"
    beam_width = 10
    
    logs_path = './logs'
    
    ckpt_path = './models/model.ckpt'
    ckpt_dir  = './models'
    
    base_model_ckpt_path = './base_models/base_model.ckpt'
    base_model_ckpt_dir  = './base_models'
    
    rouge_evaluation_dir = "./rouge"
    
    # learned tensorflow variables
    train_loss = None
    update_step = None
    train_predictions = None
    test_predictions = None


