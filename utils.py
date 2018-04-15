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

import pickle

# VOCAB dictionary processing
def getVocabDicts(params,vocab_file_path):
    
    # initialize vocabulary - add unknown_token, start and end
    vocabulary = [params.pad_token,params.unknown_token,params.sentence_start,params.sentence_end]

    # open the file and read lines
    vocab_file = open(vocab_file_path)
    vocabulary.extend(list(map(lambda x:x.strip(),vocab_file.readlines())))

    # generate indices
    vocab_index = {word:index for index,word in enumerate(vocabulary)}
    index_vocab = {index:word for index,word in enumerate(vocabulary)}
    
    return (vocab_index,index_vocab)

# map each word to its corresponding index

def mapDatasetToIndex(params,data_file_name, vocab_dict, debug = True):
    
    print("Mapping dataset %s to index. Please wait..." % data_file_name)
    
    dataset = []
    count = 0
    
    data_file = open(data_file_name)
    
    line = data_file.readline()
    
    while line != "":
        
        line = line.strip().split(" ")
        line_indices = []
        
        for word in line:
            
            if vocab_dict.get(word) != None:
                line_indices.append(vocab_dict[word])
            else:
                line_indices.append(vocab_dict[params.unknown_token])
                
        dataset.append(line_indices)
        
        line = data_file.readline()
        count += 1
        
        if debug == True:
            if count % 1000 == 0:
                print("Processing %s. Completed %d lines." % (data_file_name,count))
        
    return dataset

def create_train_and_test_datasets(params):
    
    train_in_file_path = params.data_dir + "/" + params.train_in_file
    params.train_in_dataset  = mapDatasetToIndex(params,train_in_file_path,params.story_dicts[0])
    
    train_out_file_path = params.data_dir + "/" + params.train_out_file
    params.train_out_dataset = mapDatasetToIndex(params,train_out_file_path,params.summary_dicts[0])
    
    test_in_file_path = params.data_dir + "/" + params.test_in_file
    params.test_in_dataset   = mapDatasetToIndex(params,test_in_file_path,params.story_dicts[0])
    
    test_out_file_path = params.data_dir + "/" + params.test_out_file
    params.test_out_dataset  = mapDatasetToIndex(params,test_out_file_path,params.summary_dicts[0])
    
    dev_in_file_path = params.data_dir + "/" + params.dev_in_file
    params.dev_in_dataset    = mapDatasetToIndex(params,dev_in_file_path,params.story_dicts[0])
    
    dev_out_file_path = params.data_dir + "/" + params.dev_out_file
    params.dev_out_dataset   = mapDatasetToIndex(params,dev_out_file_path,params.summary_dicts[0])

    pickle_file = params.data_dir + "/tldr_datasets.pkl"
    
    print("Writing datasets to pickle file")
    with open(pickle_file,"wb") as fp:
        pickle.dump(params.train_in_dataset,fp)
        pickle.dump(params.train_out_dataset,fp)
        pickle.dump(params.test_in_dataset,fp)
        pickle.dump(params.test_out_dataset,fp)
        pickle.dump(params.dev_in_dataset,fp)
        pickle.dump(params.dev_out_dataset,fp)
          
def restore_train_and_test_datasets(params):
          
    pickle_file = params.data_dir + "/tldr_datasets.pkl"
    
    print("Restoring datasets from pickle file")   
    with open(pickle_file,"rb") as fp:
        params.train_in_dataset = pickle.load(fp)
        params.train_out_dataset = pickle.load(fp)
        params.test_in_dataset = pickle.load(fp)
        params.test_out_dataset = pickle.load(fp)
        params.dev_in_dataset = pickle.load(fp)
        params.dev_out_dataset = pickle.load(fp)