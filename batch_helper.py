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

import numpy as np

def getBatch(params,max_time,dataset,vocab_dict,batch_type,training=True,debug=False):
    
    if debug == True:
        # sample one
        print("batch type:",batch_type)
        print("Sampling one:\n"," ".join(list(map(lambda x:story_dicts[1][x], dataset[128][0:100]))))
    
    # get the next batch_size set of data
    raw_inputs  = []
    inputs  = np.full((max_time, params.batch_size),0,dtype=np.int32)
    lengths = np.full((params.batch_size),0,dtype=np.int32)

    text_count = 0
    
    if training == True:
        batch_index = params.train_batch_index
    else:
        batch_index = params.test_batch_index
        
    for text in dataset[batch_index:batch_index+params.batch_size]:
        
        current_time = 0
        sequence = np.full((max_time),vocab_dict[params.pad_token])
        
        if batch_type == "decoder_output" or batch_type == "decoder_input":
            word_count = (len(text)+1) if (len(text)+1) < max_time else max_time
        else:
            word_count = len(text) if len(text) < max_time else max_time
            
        lengths[text_count] = word_count
        
        # set start and end word positions
        word_start = 0
        word_end = word_count
        
        # are we working with decoder input/output batches then populate start/end of sentence
        # markers and adjust the range indices appropriately
        if batch_type == "decoder_input":
            sequence[0] = vocab_dict[params.sentence_start]
            word_start = 1
        elif batch_type == "decoder_output":
            sequence[word_count-1] = vocab_dict[params.sentence_end]
            word_end= word_count-1
        
        origIdx = 0
        for wordIdx in range(word_start,word_end):
            sequence[wordIdx] = int(text[origIdx])
            origIdx += 1
            
        # add to raw inputs
        if batch_type == "encoder_input":
            raw_inputs.append([params.story_dicts[1][x] for x in sequence])
        else:
            raw_inputs.append([params.summary_dicts[1][x] for x in sequence])
            
        # add to our batch_size  matrix
        inputs[:,text_count] = sequence

        if debug == True:
            if batch_count == 0:
                print("batch_type:",batch_type)
                print("Text count:",text_count)
                print("sequence:",sequence)
                if batch_type == "decoder_input" or batch_type == "decoder_output":
                    print("sequence_reverse using summary:\n",[params.summary_dicts[1][x] for x in sequence])
                else:
                    print("sequence_reverse using story:\n",[params.story_dicts[1][x] for x in sequence])
                print("inputs column in max_time major format:", inputs[:,text_count])
                debug = False
            
        # increment batch count
        text_count += 1
        
    if len(raw_inputs) == 0:
        
        print("getBatch encountered zero raw_inputs in return")
        print("batch_type:",batch_type)
        print("test_batch_index:",test_batch_index)
        print("raw_inputs:\n",raw_inputs)
        print("inputs:\n",inputs)
        print("lengths:\n",lengths)
        
    return (raw_inputs,inputs,lengths)

def getNextBatch(params,story_dataset,summary_dataset,training=True):
    
    # reset batch index if we reached the end 
    if training == True:
        if params.train_batch_index + params.batch_size > len(summary_dataset):
            return None
    else:
        if params.test_batch_index + params.batch_size > len(summary_dataset):
            print("About to return None for test_batch. test_batch_index:%d,params.batch_size:%d,len(summary_dataset:%d)"
                 % (params.test_batch_index,params.batch_size,len(summary_dataset)))
            return None
            
    # get encoder input
    raw_enc_in,enc_in,enc_in_lengths, = getBatch(params,params.encoder_max_time,story_dataset,params.story_dicts[0],"encoder_input",training=training)
    
    # get decoder input
    raw_dec_in,dec_in,dec_in_lengths = getBatch(params,params.decoder_max_time,summary_dataset,params.summary_dicts[0],"decoder_input",training=training)
    
    # get decoder output
    raw_dec_out,dec_out,dec_out_lengths = getBatch(params,params.decoder_max_time,summary_dataset,params.summary_dicts[0],"decoder_output",training=training)
        
    # increment batch_size
    if training == True:
        params.train_batch_index += params.batch_size
    else:
        params.test_batch_index += params.batch_size

    # tweak decoder outputs to be a function of the max length
    dec_in = dec_in[:max(dec_in_lengths),:]
    dec_out = dec_out[:max(dec_out_lengths),:]
    
    if len(raw_enc_in) == 0:
        
        print("getNextBatch encountered zero raw_inputs in return")
        print("raw_enc_in:\n",raw_enc_in)
        print("enc_in:\n",enc_in)
        print("enc_in_lengths:\n",enc_in_lengths)
        print("raw_dec_in:\n",raw_dec_in)
        print("dec_in:\n",dec_in)
        print("dec_in_lengths:\n",dec_in_lengths)
        print("raw_dec_out:\n",raw_dec_out)
        print("dec_out:\n",dec_out)
        print("dec_out_lengths:\n",dec_out_lengths)
    
    return (raw_enc_in,enc_in,enc_in_lengths,raw_dec_in,dec_in,dec_in_lengths,raw_dec_out,dec_out,dec_out_lengths)