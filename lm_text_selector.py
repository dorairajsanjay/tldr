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

# This module implements language model based selection of the summary with the highest likelihood

import nltk
from nltk.corpus import brown
import numpy as np

class LMTextSelector:
    
    def __init__(self):
        
        brown_bigrams = nltk.bigrams(brown.words())

        self.cfd = nltk.ConditionalFreqDist(brown_bigrams)
        self.bigrams_count = len(brown.words())
        
    def getBest(self,summaries):
        
        '''
        Input:
             - summaries is a list of all possible summaries
            - each summary is a string of words
            - we use the bigram probability to compute the best summary
        Output:
            best sentence (string)
        '''
        
        best_summary = None
        best_log_p = 0
        best_summary_index = None
        
        #print("LM Selector: Number of summaries:%d, Total Brown Bigram Count:%d" % (len(summaries),self.bigrams_count))
        #print("Count of bigrams:",self.bigrams_count)
        summary_index = 0
        for summary in summaries:
            
            # obtain all bigrams
            bigrams = nltk.bigrams(summary.split())
            
            # compute the score for each bigram
            total_log_p = 0
            for b in bigrams:
                frequency = self.cfd[b[0]][b[1]]

                temp = frequency/self.bigrams_count
                #print("temp is:",temp)
                if temp == 0:
                    log_p = 0
                    #print("Forcing log_p to 0 since frequency/self.bigrams_count is 0")
                else:
                    log_p = np.log(temp)
                
                total_log_p += log_p
                
            if best_summary == None or total_log_p > best_log_p:
                best_summary = summary
                best_log_p = total_log_p
                best_summary_index = summary_index
                
            summary_index += 1
                
        return best_summary, best_summary_index 
            
            
        
     