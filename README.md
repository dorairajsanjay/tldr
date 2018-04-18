# tldr

tldr is an experimental text summarizer that uses sequence to sequence neural machine learning models and an n-gram language model similar to other traditional noisy channel approaches. 

The language model uses the Brown corpus. It compares a subset of top Beam search scores and their bigram likelihood scores and picks the one with the highest bigram likelihood.

See pip_env_list.txt and conda_env_list.txt for a list of all installed packages in the conda environment.

The network diagram is shown below

![alt text](https://github.com/dorairajsanjay/tldr/blob/master/tldr_network_diagram.png)

### To parse the CNN stories, generate vocab and dump into a pickle file for later dataset generation

You can download CNN data from https://cs.nyu.edu/~kcho/DMQA/

* Note the below directory for the downloaded files directory = 'data/cnn/stories/'

python tldr_main.py --parse_cnn_stories

### To regenerate dataset. This is required to be done once. Subsequent attempts load from a pickle file
python tldr_main.py --regenerate_dataset

### To train and batch validate
python tldr_main.py 

There are several options here, but the key are two options
```
python tldr_main.py --inference_style "beam_search"

This is the default option.
It does inference using beam_search and also uses the language model for surfacing up the best summary according to the language model.
```
and
```
python tldr_main.py --inference_style  "greedy_search"

This option simply does a greedy_search at the decoder output and displays the corresponding summary
```

### To perform inference only on a single story fed at a time

```
python tldr_main.py --mode="inference_only"
If inference_only is enabled, add stories to inference.in in the format <transaction_id>,<story>. The result will be added to inference.out in the format <transaction_id>,<summary_id>. The transaction_id is used to correlate the summary back to the input story.
```

All models are checkpointed so you should not have to restart training from the beginning. Note that previous models get saved in a separate folder, so they may accumalate and take up space. Make sure to delete those folders in that case.

#### Below are defaults used by the model. They can be changed using the syntax below

```
Training properties:
batch size             :128
vocab size             :40000
hidden units           :128
embedding size         :128
max grad norm          :1
learning rate          :0.0001
encoder max time       :300
decoder max time       :20
keep probability       :0.8000

Inference properties:
inference_style        :beam_search
beam width             :10
lm beam width          :3
mode                   :inference_only

Environment properties:
model directory        :./models
ignore checkpoints     :False
data directory         :./data
parse cnn stories      :False
regenerate dataset     :False
story vocab file       :vocab.in
summary vocab file     :vocab.out
training stories file  :train.in
training summaries file:train.out
test stories file      :test.in
test summaries file    :test.out
inference input file   :inference.in
inference out file     :inference.out

Display properties:
max display len        :12
max summary len        :20
```
### Syntax
```
usage: tldr_main.py [-h] [--batch_size BATCH_SIZE] [--vocab_size VOCAB_SIZE]
                    [--hidden_units HIDDEN_UNITS]
                    [--embedding_size EMBEDDING_SIZE]
                    [--max_grad_norm MAX_GRAD_NORM]
                    [--learning_rate LEARNING_RATE] [--keep_prob KEEP_PROB]
                    [--model_dir MODEL_DIR] [--ignore_checkpoint]
                    [--data_dir DATA_DIR] [--parse_cnn_stories]
                    [--regenerate_dataset]
                    [--story_vocab_file STORY_VOCAB_FILE]
                    [--summary_vocab_file SUMMARY_VOCAB_FILE]
                    [--train_in_file TRAIN_IN_FILE]
                    [--train_out_file TRAIN_OUT_FILE]
                    [--test_in_file TEST_IN_FILE]
                    [--test_out_file TEST_OUT_FILE]
                    [--inference_in_file INFERENCE_IN_FILE]
                    [--inference_out_file INFERENCE_OUT_FILE]
                    [--inference_style INFERENCE_STYLE]
                    [--beam_width BEAM_WIDTH] [--lm_beam_width LM_BEAM_WIDTH]
                    [--mode MODE] [--max_display_len MAX_DISPLAY_LEN]
                    [--max_summary_len MAX_SUMMARY_LEN]
                    [--encoder_max_time ENCODER_MAX_TIME]
                    [--decoder_max_time DECODER_MAX_TIME]

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
  --keep_prob KEEP_PROB
                        keep probability. this is (1 - the drop-out
                        probability)
  --model_dir MODEL_DIR
                        path to saved models
  --ignore_checkpoint   ignore existing checkpoints for restore
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
  --inference_in_file INFERENCE_IN_FILE
                        inference stories file, valid only when inference_only
                        is enabled
  --inference_out_file INFERENCE_OUT_FILE
                        inference summaries file, valid only when
                        inference_only is enabled
  --inference_style INFERENCE_STYLE
                        type of inference - beam_search or greedy_search
  --beam_width BEAM_WIDTH
                        beam search width or beam size
  --lm_beam_width LM_BEAM_WIDTH
                        beam search outputs for validation against the
                        language model
  --mode MODE           options are train_inference and inference_only. If
                        inference_only is enabled, add stories to inference.in
                        in the format <story_id> <story>. The result will be
                        added to inference.out in the format <story_id>
                        <summary>
  --max_display_len MAX_DISPLAY_LEN
                        number of words to pick when displaying summary
  --max_summary_len MAX_SUMMARY_LEN
                        number of words to pick for the summary
  --encoder_max_time ENCODER_MAX_TIME
                        number of steps to unroll for the encoder
  --decoder_max_time DECODER_MAX_TIME
                        number of steps to unroll for the summary
  ```  
  #### Features
  
  1. RNN/LSTM based sequence to sequence network
  2. General utility/debugging features - tensorboard, model persistence
  
  #### Models
  
  ##### tldr_model_base.py
  
  Implements basic seq2seq summarization. A simple LSTM encoder and decoder
  
  ##### tldr_model.py
  
  Slightly advanced seq2seq summarization featuring Dropout and Bahdanau attention and Beam Search
  
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
