# tldr
text summarizer

## To parse the CNN stories, generate vocab and dump into a pickle file for later dataset generation
python tldr_main.py --parse_cnn_stories

## To regenerate dataset. This is required to be done once. Subsequent attempts load from a pickle file
python tldr_main.py --regenerate_dataset

## To train and validate 
python tldr_main.py 

### Below are defaults used by the model. They can be changed using the syntax below

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

## Syntax

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
                        
                        
