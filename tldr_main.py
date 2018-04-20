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
        tldr_model.run(params)
 
if __name__ == "__main__":
    
    params = model_params.ModelParams()

    # processing input arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size",type=int,default=64,help="batch size")
    parser.add_argument("--vocab_size",type=int,default=40000,help="vocabulary size to generate when generating the full dataset")
    parser.add_argument("--hidden_units",type=int,default=128,help="number of LSTM hidden units")
    parser.add_argument("--embedding_size",type=int,default=128,help="number of embedding dimensions")
    parser.add_argument("--max_grad_norm",type=int,default=1,help="max grad norm. typically integer 1 through 5")
    parser.add_argument("--learning_rate",type=float,default=0.001,help="learning rate")
    parser.add_argument("--keep_prob",type=float,default=0.8,help="keep probability. this is (1 - the drop-out probability)")

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
    parser.add_argument("--inference_in_file",default="inference.in",help="inference stories file, valid only when inference_only is enabled")
    parser.add_argument("--inference_out_file",default="inference.out",help="inference summaries file, valid only when inference_only is enabled")
    
    parser.add_argument("--inference_style",default="beam_search",help="type of inference - beam_search or greedy_search")
    parser.add_argument("--beam_width",type=int,default=10,help="beam search width or beam size")
    parser.add_argument("--lm_beam_width",type=int,default=3,help="beam search outputs for validation against the language model")
    parser.add_argument("--mode",default="train_inference",help="options are train_inference and inference_only. If inference_only is enabled, add stories to inference.in in the format <story_id> <story>. The result will be added to inference.out in the format <story_id> <summary>")
    
    parser.add_argument("--max_display_len",type=int,default=20,help="number of words to pick when displaying summary")
    parser.add_argument("--max_summary_len",type=int,default=20,help="number of words to pick for the summary")
    parser.add_argument("--encoder_max_time",type=int,default=300,help="number of steps to unroll for the encoder")
    parser.add_argument("--decoder_max_time",type=int,default=20,help="number of steps to unroll for the summary")
    
    
    args = parser.parse_args()
    
    # display arguments
    print("Running program with below arguments:\n")
    
    params_str = ""
    
    params_str += ("\nTraining properties:")
    params_str += "\nbatch size             :%d"    % (args.batch_size); params.batch_size = args.batch_size
    params_str += ("\nvocab size             :%d"    % args.vocab_size); params.vocab_size = args.vocab_size    
    params_str += ("\nhidden units           :%d"    % args.hidden_units); params.hidden_units = args.hidden_units
    params_str += ("\nembedding size         :%d"    % args.embedding_size); params.embedding_size = args.embedding_size
    params_str += ("\nmax grad norm          :%d"    % args.max_grad_norm); params.max_grad_norm = args.max_grad_norm
    params_str += ("\nlearning rate          :%0.4f" % args.learning_rate); params.learning_rate = args.learning_rate
    params_str += ("\nencoder max time       :%d"    % args.encoder_max_time); params.encoder_max_time = args.encoder_max_time     
    params_str += ("\ndecoder max time       :%d"    % args.decoder_max_time); params.decoder_max_time = args.decoder_max_time    
    params_str += ("\nkeep probability       :%0.4f" % args.keep_prob); params.keep_prob = args.keep_prob  
    
    params_str += ("\n\nInference properties:")
    params_str += ("\ninference_style        :%s"    % args.inference_style); params.inference_style = args.inference_style     
    params_str += ("\nbeam width             :%d"    % args.beam_width); params.beam_width = args.beam_width     
    params_str += ("\nlm beam width          :%d"    % args.lm_beam_width); params.lm_beam_width = args.lm_beam_width       
    params_str += ("\nmode                   :%s"    % args.mode); params.mode = args.mode    
    
    params_str += ("\n\nEnvironment properties:")
    params_str += ("\nmodel directory        :%s"    % args.model_dir); params.model_dir = args.model_dir
    params_str += ("\nignore checkpoints     :%s"    % args.ignore_checkpoint); params.ignore_checkpoint = args.ignore_checkpoint
    params_str += ("\ndata directory         :%s"    % args.data_dir); params.data_dir = args.data_dir
    params_str += ("\nparse cnn stories      :%s"    % args.parse_cnn_stories); params.parse_cnn_stories = args.parse_cnn_stories
    params_str += ("\nregenerate dataset     :%s"    % args.regenerate_dataset); params.regenerate_dataset = args.regenerate_dataset           
    params_str += ("\nstory vocab file       :%s"    % args.story_vocab_file); params.story_vocab_file = args.story_vocab_file
    params_str += ("\nsummary vocab file     :%s"    % args.summary_vocab_file); params.summary_vocab_file = args.summary_vocab_file
    params_str += ("\ntraining stories file  :%s"    % args.train_in_file); params.train_in_file = args.train_in_file
    params_str += ("\ntraining summaries file:%s"    % args.train_out_file); params.train_out_file = args.train_out_file
    params_str += ("\ntest stories file      :%s"    % args.test_in_file); params.test_in_file = args.test_in_file     
    params_str += ("\ntest summaries file    :%s"    % args.test_out_file); params.test_out_file = args.test_out_file  
    params_str += ("\ninference input file   :%s"    % args.inference_in_file); params.inference_in_file = args.inference_in_file
    params_str += ("\ninference out file     :%s"    % args.inference_out_file); params.inference_out_file = args.inference_out_file 
    params_str += ("\n\nDisplay properties:")
    params_str += ("\nmax display len        :%d"    % args.max_display_len); params.max_display_len = args.max_display_len     
    params_str += ("\nmax summary len        :%d"    % args.max_summary_len); params.max_summary_len = args.max_summary_len     

    params_str += "\n"
    params_str += ("#"*80)
    params_str += "\n"
    
    # display params
    print(params_str)
    
    # write params to file
    print("Writing model params to :",params.params_file)
    open(params.params_file,"w").write(params_str)
    
    # Run program
    run_program(params)
    

    
    
    
                    

                    
                    




