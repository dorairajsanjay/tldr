# tldr

tldr is an experimental text summarizer that uses sequence to sequence neural machine learning models.

### To parse the CNN stories, generate vocab and dump into a pickle file for later dataset generation

You can download CNN data from https://cs.nyu.edu/~kcho/DMQA/

* Note the below directory for the downloaded files directory = 'data/cnn/stories/'

python tldr_main.py --parse_cnn_stories

### To regenerate dataset. This is required to be done once. Subsequent attempts load from a pickle file
python tldr_main.py --regenerate_dataset

### To train and validate 
python tldr_main.py 

#### Below are defaults used by the model. They can be changed using the syntax below

```
batch size             :128
vocab size             :20000
hidden units           :128
embedding size         :128
max grad norm          :1
learning rate          :0.0010
model directory        :./models
restore model path     :False
data directory         :./data
parse cnn stories      :False
regenerate dataset     :False
story vocab file       :vocab.in
summary vocab file     :vocab.out
training stories file  :train.in
training summaries file:train.out
test stories file      :test.in
test summaries file    :test.out
```
### Syntax
```
usage: tldr_main.py [-h] [--batch_size BATCH_SIZE] [--vocab_size VOCAB_SIZE]
                    [--hidden_units HIDDEN_UNITS]
                    [--embedding_size EMBEDDING_SIZE]
                    [--max_grad_norm MAX_GRAD_NORM]
                    [--learning_rate LEARNING_RATE] [--model_dir MODEL_DIR]
                    [--restore_saved_model] [--data_dir DATA_DIR]
                    [--parse_cnn_stories] [--regenerate_dataset]
                    [--story_vocab_file STORY_VOCAB_FILE]
                    [--summary_vocab_file SUMMARY_VOCAB_FILE]
                    [--train_in_file TRAIN_IN_FILE]
                    [--train_out_file TRAIN_OUT_FILE]
                    [--test_in_file TEST_IN_FILE]
                    [--test_out_file TEST_OUT_FILE]

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        batch size
  --vocab_size VOCAB_SIZE
                        vocabulary size to generate when generating the full
                        dataset
  --hidden_units HIDDEN_UNITS
                        number of LSTM hidden units
  --embedding_size EMBEDDING_SIZE
                        number of embedding dimensions
  --max_grad_norm MAX_GRAD_NORM
                        max grad norm. typically integer 1 through 5
  --learning_rate LEARNING_RATE
                        learning rate
  --model_dir MODEL_DIR
                        path to saved models
  --restore_saved_model
                        restore saved model
  --data_dir DATA_DIR   path to data
  --parse_cnn_stories   parses CNN dataset. Generates stories and vocab files
                        in <data_dir>. Expects unzipped cnn stores in data
                        directory of the form <data_dir>/cnn/stories/*.story
  --regenerate_dataset  regenerates training and test data from pickle file in
                        data directory
  --story_vocab_file STORY_VOCAB_FILE
                        story vocabulary file
  --summary_vocab_file SUMMARY_VOCAB_FILE
                        summary vocabulary file
  --train_in_file TRAIN_IN_FILE
                        training stories file
  --train_out_file TRAIN_OUT_FILE
                        training summaries file
  --test_in_file TEST_IN_FILE
                        test stories file
  --test_out_file TEST_OUT_FILE
                        test summaries file
                        
  ```  
  #### Features
  
  1. RNN/LSTM based sequence to sequence network
  2. General utility/debugging features - tensorboard, model persistence
  
  #### Models
  
  ##### tldr_model_base.py
  
  Implements basic seq2seq summarization. A simple LSTM encoder and decoder
  
  ##### tldr_model.py
  
  Slightly advanced seq2seq summarization featuring Dropout and Bahdanau attention. 
  
  #### Modules
  
  * tldr_main.py - Entry point
  * tldr_model.py - sequence 2 sequence model
  * dataset_helper.py - helper module to parse CNN stories and create dataset
  * model_params.py - contains all variables used in the model
  * utils.py - common utility functions
  * batch_helper.py - helper for batch processing
  
  ### Running ROUGE evaluation
  
  Files for ROUGE evaluation are stored in ./rouge
  
  There are two files - hypothesis.txt and reference.txt
  
  In order to obtain the use rouge, you will need to download the ROUGE package
  
  You can do this using the below command
  
  ```
  sudo pip3 install rouge
  ```
  
  Once you have downloaded this, you can compute the average ROUGE score using the below command. The output is a JSON file that has ROUGE metrics
  
  ```
  rouge -a rouge/hypothesis.txt rouge/reference.txt
  ```
  
  #### Viewing Tensorboard logs
  
  ```
  tensorboard --logdir=./logs
  ```
  
  #### Folder Structure
  ```
  (tensorflow_p36plus) ubuntu@ip-172-31-16-7:~/dev$ tree -d tldr
tldr
├── data
│   └── cnn
│       └── stories
├── logs
├── models
└── __pycache__
```
