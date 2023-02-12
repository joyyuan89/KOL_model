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
speeches_data['date'] = pd.to_datetime(speeches_data['date'], format="%d/%m/%Y")
speeches_data.set_index('date', inplace=True)

# randomly select 10 row for test
# df_sample = speeches_data.sample(n = 3)
df_sample = speeches_data

#%% Vectorizer
from keyphrase_vectorizers import KeyphraseCountVectorizer
pos_pattern = '<J.*>*<N.*>+'
vectorizer = KeyphraseCountVectorizer(pos_pattern=pos_pattern)

#%% Embedding model
from sentence_transformers import SentenceTransformer,util

def embedding(text):
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    doc_embeddings = embedder.encode(text)

    return doc_embeddings

#%% Cosine similarity
from numpy.linalg import norm

def cosine_similarity_function(vec_1, vec_2):
    value = np.dot(vec_1, vec_2.T)/(norm(vec_1)*norm(vec_2))
    return value

#%% Variables
search_word = 'covid'

#%% Main body
import time
start_time = time.time()
print("program running")

Search_word_embedding = embedding(search_word)

df_keywords = df_sample.copy()

# df_keywords["keywords"] = df_keywords["text"].apply(lambda x: keyBERT_keyword(x ,10))
df_keywords["text_embedding"] = df_keywords["text"].apply(lambda x: embedding(x))

# df_keywords["keywords_embedding"] = df_keywords["keywords"].apply(lambda x: keyBERT_embedding(x))
df_keywords[search_word] = df_keywords["text_embedding"].apply(lambda x: cosine_similarity_function(x, Search_word_embedding))

print("program completed")
print("--- %s min ---" % (time.time() - start_time))/60

#%% Post processing

date_range = pd.date_range(start=df_keywords.index.min(), end=df_keywords.index.max(),freq='D')
date_range = date_range.to_frame()
df_output = pd.merge(df_keywords, date_range, left_index=True, right_index=True, how='outer')
df_output.fillna(0,inplace=True)
df_output["value"] = df_output[search_word].rolling(window=30, min_periods=1).sum()

#%% Plot
import matplotlib.pyplot as plt

x = df_output[0]
y = df_output["value"]

plt.xlabel("Date")
plt.ylabel(search_word+" value")
plt.title(search_word+" trend")

plt.plot(x, y)
plt.show()

#%% Download
df_output.to_excel("df_output.xlsx")
df_keywords.to_excel("df_keywords.xlsx")
