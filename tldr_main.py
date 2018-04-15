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

# This module is the entry point to the tldr summarization program

import sys
import argparse

import model_params
import dataset_helper
import tldr_model

def run_program(params):
    
    if params.parse_cnn_stories:
        dataset_helper.parse_cnn_stories(params)   
    else:
        tldr_model.train(params)
 
if __name__ == "__main__":
    
    params = model_params.ModelParams()

    # processing input arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size",type=int,default=128,help="batch size")
    parser.add_argument("--vocab_size",type=int,default=40000,help="vocabulary size to generate when generating the full dataset")
    parser.add_argument("--hidden_units",type=int,default=128,help="number of LSTM hidden units")
    parser.add_argument("--embedding_size",type=int,default=128,help="number of embedding dimensions")
    parser.add_argument("--max_grad_norm",type=int,default=1,help="max grad norm. typically integer 1 through 5")
    parser.add_argument("--learning_rate",type=float,default=0.001,help="learning rate")

    parser.add_argument("--model_dir",default="./models",help="path to saved models")
    parser.add_argument("--ignore_checkpoint", action='store_true',help="ignore existing checkpoints for restore")

    parser.add_argument("--data_dir",default="./data",help="path to data")

    parser.add_argument("--parse_cnn_stories",action='store_true',help="parses CNN dataset. Generates stories and vocab files in <data_dir>. Expects unzipped cnn stores in data directory of the form <data_dir>/cnn/stories/*.story")
    parser.add_argument("--regenerate_dataset",action='store_true',help="regenerates training and test data from pickle file in data directory")

    parser.add_argument("--story_vocab_file",default="vocab.in",help="story vocabulary file")
    parser.add_argument("--summary_vocab_file",default="vocab.out",help="summary vocabulary file")
    parser.add_argument("--train_in_file",default="train.in",help="training stories file")
    parser.add_argument("--train_out_file",default="train.out",help="training summaries file")
    parser.add_argument("--test_in_file",default="test.in",help="test stories file")
    parser.add_argument("--test_out_file",default="test.out",help="test summaries file")
    
    args = parser.parse_args()
    
    # display arguments
    print("Running program with below arguments:\n")
    
    print("batch size             :%d"    % args.batch_size); params.batch_size = args.batch_size
    print("vocab size             :%d"    % args.vocab_size); params.vocab_size = args.vocab_size    
    print("hidden units           :%d"    % args.hidden_units); params.hidden_units = args.hidden_units
    print("embedding size         :%d"    % args.embedding_size); params.embedding_size = args.embedding_size
    print("max grad norm          :%d"    % args.max_grad_norm); params.max_grad_norm = args.max_grad_norm
    print("learning rate          :%0.4f" % args.learning_rate); params.learning_rate = args.learning_rate
    print("model directory        :%s"    % args.model_dir); params.model_dir = args.model_dir
    print("ignore checkpoints     :%s"    % args.ignore_checkpoint); params.ignore_checkpoint = args.ignore_checkpoint
    print("data directory         :%s"    % args.data_dir); params.data_dir = args.data_dir
    print("parse cnn stories      :%s"    % args.parse_cnn_stories); params.parse_cnn_stories = args.parse_cnn_stories
    print("regenerate dataset     :%s"    % args.regenerate_dataset); params.regenerate_dataset = args.regenerate_dataset
                        
    print("story vocab file       :%s"    % args.story_vocab_file); params.story_vocab_file = args.story_vocab_file
    print("summary vocab file     :%s"    % args.summary_vocab_file); params.summary_vocab_file = args.summary_vocab_file
    print("training stories file  :%s"    % args.train_in_file); params.train_in_file = args.train_in_file
    print("training summaries file:%s"    % args.train_out_file); params.train_out_file = args.train_out_file
    print("test stories file      :%s"    % args.test_in_file); params.test_in_file = args.test_in_file     
    print("test summaries file    :%s"    % args.test_out_file); params.test_out_file = args.test_out_file  
    
    print("#"*80)
    
    # Run program
    run_program(params)
    

    
    
    
                    

                    
                    




