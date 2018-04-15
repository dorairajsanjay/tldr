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

#You can download CNN data from https://cs.nyu.edu/~kcho/DMQA/
#Note the below directory for the downloaded files directory = 'data/cnn/stories/'

#################################################################################
# A good part of this code has been adapted from 
# https://machinelearningmastery.com/prepare-news-articles-# text-summarization/ 
# for the purpose of loading data from CNN dataset.
#################################################################################

from os import listdir
import string
import nltk
import re
from collections import Counter
from pickle import dump

# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, encoding='utf-8')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

# split a document into news story and highlights
def split_story(doc):
    # find first highlight
    index = doc.find('@highlight')
    # split into story and highlights
    story, highlights = doc[:index], doc[index:].split('@highlight')
    # strip extra white space around each highlight
    highlights = [h.strip() for h in highlights if len(h) > 0]
    return story, highlights

# load all stories in a directory
def load_stories(directory):
    stories = list()
    for name in listdir(directory):
        filename = directory + '/' + name
        # load document
        doc = load_doc(filename)
        # split into story and highlights
        story, highlights = split_story(doc)
        # store
        stories.append({'story':story, 'highlights':highlights})
    return stories

# clean a list of lines
def clean_lines(lines):
    cleaned = list()
    
    # prepare a translation table to remove punctuation
    table = str.maketrans('', '', string.punctuation)
    for line in lines:
        # strip source cnn office if it exists
        index = line.find('(CNN) -- ')
        if index > -1:
            line = line[index+len('(CNN)'):]
        # tokenize on white space
        line = line.split()
        # convert to lower case
        line = [word.lower() for word in line]
        # remove punctuation from each token
        line = [w.translate(table) for w in line]
        # remove tokens with numbers in them
        line = [word for word in line if word.isalpha()]
        # store as string
        cleaned.append(' '.join(line))
    # remove empty strings
    cleaned = [c for c in cleaned if len(c) > 0]
    return cleaned

def getData(stories,train_pct,dev_pct,test_pct):
    
    '''
    Input:
    
    stories  : list of stories. Each element is a dictionar with two keys and corresponding values.
               key1 is "highlight" and key2 is story. 
               "highlight" is a list of highlights or summary of the story spread across multiple lines
               "story" is the news story spread across multiple lines
    train_pct: floating point number. Percentage to dedicate for training
    dev_pct  : floating point number. Percentage to dedicate for dev testset
    test_pct : floating point number. Percentage to dedicate for testing
    
    Output:
    
    train_in, train_out, dev_in, dev_out, test_in, test_out: 
                corresponding to training and test input and output sentences.
                All sentences are squashed to a single line
    '''
    
    data_in = []
    data_out = []
    
    allHighlights = False
    
    for story in stories:
        
        if allHighlights == False:
            summary = story["highlights"][0]
        else:
            summary = ".".join(story["highlights"])
        details = ".".join(story["story"])
        
        # remove empty lines
        if details == "":
            continue
        
        # add to dataset
        data_in.append(details)
        data_out.append(summary)

    # splice list
    train_end  = int(len(data_in) * abs(float(train_pct)/100))
    dev_begin  = train_end # to adjust for python being zero based
    dev_end    = int(len(data_in) * abs(float(train_pct+dev_pct)/100))
    test_begin = dev_end
    
    train_in  = data_in[0:train_end]
    train_out = data_out[0:train_end]
    
    dev_in = data_in[dev_begin:dev_end]
    dev_out = data_out[dev_begin:dev_end]
    
    test_in = data_in[test_begin:]
    test_out = data_out[test_begin:]
    
    return train_in, train_out, dev_in, dev_out, test_in, test_out

def restore_dataset_from_file(params):
    
    import pickle
    
    pickle_file = params.data_dir + "/cnn_dataset.pkl"

    pFile = open(pickle_file, 'rb')
    pFile.seek(0)

    stories = pickle.load(pFile)
    print('Loaded Stories %d' % len(stories))
    
def create_dataset_files(params,stories):
    
    # invoke method to get training, dev and test set at 90:5:5 proporation
    train_in, train_out, dev_in, dev_out, test_in, test_out =\
                                                getData(stories,90,5,5)
    print("train_in:%d, train_out:%d, dev_in:%d, dev_out:%d, test_in:%d, \
          test_out:%d" % (len(train_in), len(train_out), len(dev_in), \
          len(dev_out),len(test_in), len(test_out)))    
    
    # write data out to file
    with open(params.data_dir + "/train.in","w") as outfile:
        outfile.write('\n'.join(train_in))

    with open(params.data_dir + "/train.out","w") as outfile:
        outfile.write('\n'.join(train_out))

    with open(params.data_dir +"/dev.in","w") as outfile:
        outfile.write('\n'.join(dev_in))

    with open(params.data_dir +"/dev.out","w") as outfile:
        outfile.write('\n'.join(dev_out))

    with open(params.data_dir +"/test.in","w") as outfile:
        outfile.write('\n'.join(test_in))

    with open(params.data_dir +"/test.out","w") as outfile:
        outfile.write('\n'.join(test_out))
        
def create_stories_vocabulary(params,stories):
    stories_dataset = [item["story"] for item in stories]
    
    stories_dataset_final = list(map(lambda x: ".".join(x),stories_dataset))
    
    # write these <vocab-size> into vocab file
    vocabSize = params.vocab_size
    vocabFileName = params.data_dir + "/vocab.in"

    all_tokens = []

    count = 1
    for story in stories_dataset_final:

        tokens = nltk.word_tokenize(story)

        all_tokens.extend(tokens)

        count += 1
        if count%2000 == 0:
            print("Tokenization in progress. Completed:%d. Remaining:%d" % (count,len(stories_dataset_final)-count))

    counts = Counter(all_tokens)

    # get most common words
    common_words = counts.most_common(vocabSize)

    # write to output file
    print("Writing to file:",vocabFileName)
    print("Common words count:",len(common_words))
    with open(vocabFileName,"w") as vocabFile:
        for i in range(0,vocabSize):
            vocabFile.write(common_words[i][0])
            vocabFile.write("\n")
            
def create_summaries_vocabulary(params,stories):
    
    highlights_dataset = [item["highlights"][0] for item in stories]
    
    # write these <vocab-size> into vocab file
    vocabSize = params.vocab_size
    vocabFileName = params.data_dir + "/vocab.out"

    all_tokens = []

    for highlight in highlights_dataset:

        tokens = nltk.word_tokenize(highlight)

        all_tokens.extend(tokens)

    counts = Counter(all_tokens)

    # get most common words
    common_words = counts.most_common(vocabSize)

    # write to output file
    # write to output file
    print("Writing to file:",vocabFileName)
    print("Common words count:",len(common_words))
    with open(vocabFileName,"w") as vocabFile:
        for i in range(0,vocabSize):
            vocabFile.write(common_words[i][0])
            vocabFile.write("\n")
    
def parse_cnn_stories(params):
    
    print("Parsing CNN stories and generating a pickle file for the stories and vocabulary files for stories and summaries...this may take a while...")
    
    # load stories
    directory = params.data_dir + '/cnn/stories'
    stories = load_stories(directory)
    
    print('Loaded Stories %d' % len(stories))

    # clean stories
    for example in stories:
        example['story'] = clean_lines(example['story'].split('\n'))
        example['highlights'] = clean_lines(example['highlights'])
    
    # creating dataset files for training, testing and dev
    print("Creating dataset files")
    create_dataset_files(params,stories)
    
    # generate a vocab file for the training data
    print("Creating stories vocabulary...")
    create_stories_vocabulary(params,stories)
    
    # generate a vocab file for the test data
    print("Creating summaries vocabulary...")
    create_summaries_vocabulary(params,stories)
    
    # save to file
    pickle_file = params.data_dir + "/cnn_dataset.pkl"
    print("Writing to pickle file:",pickle_file)

    dump(stories, open(pickle_file, 'wb'))
        

        
        
   
