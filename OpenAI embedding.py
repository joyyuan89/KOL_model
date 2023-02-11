#%% Setup
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 11:12:50 2023

@author: jakewen
"""

# libraries
import pandas as pd
import numpy as np
import os

# dir
work_dir = os.getcwd()

# load data
input_path = "/Users/jakewen/Desktop/Github/KOL_model/INPUT/central_bank_speech/all_speeches.csv"

speeches_data = pd.read_csv(input_path)

# randomly select 10 row for test
df_sample = speeches_data.sample(n = 5)

#%% Vectorizer
from keyphrase_vectorizers import KeyphraseCountVectorizer
pos_pattern = '<J.*>*<N.*>+'
vectorizer = KeyphraseCountVectorizer(pos_pattern=pos_pattern)

#%% KeyBERT
from keybert import KeyBERT

def keyBERT_keyword(text,top_n):
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(
                      text,
                      vectorizer=vectorizer,
                      # use_maxsum=True,
                       # use_mmr=True, 
                      # diversity=0.7,
                      # nr_candidates=20, 
                      top_n=top_n)
  
    li_keywords = [pair[0] for pair in keywords] # keywords list without similarity value

    return li_keywords

def keyBERT_embedding(text):
    kw_model = KeyBERT()
    doc_embeddings, word_embeddings  = kw_model.extract_embeddings(
        text,
        candidates=None, 
        stop_words='english', 
        min_df=1, 
        vectorizer=None)

    return doc_embeddings

#%% Cosine similarity
from numpy.linalg import norm

def cosine_similarity_function(vec_1, vec_2):
    value = np.dot(vec_1, vec_2.T)/(norm(vec_1)*norm(vec_2))
    return value[0][0]

#%% Variables

Search_word = 'inflation'

#%% Main body

import time
start_time = time.time()

Search_word_embedding = keyBERT_embedding(Search_word)

df_keywords = df_sample.copy()

df_keywords["keywords"] = df_keywords["text"].apply(lambda x: keyBERT_keyword(x ,10))
df_keywords["text_embedding"] = df_keywords["text"].apply(lambda x: keyBERT_embedding(x))
df_keywords["keywords_embedding"] = df_keywords["keywords"].apply(lambda x: keyBERT_embedding(x))
df_keywords["Cosine_similarity_article"] = df_keywords["text_embedding"].apply(lambda x: cosine_similarity_function(x, Search_word_embedding))

print("--- %s seconds ---" % (time.time() - start_time))
