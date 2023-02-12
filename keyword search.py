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
import re
import ast

# dir
work_dir = os.getcwd()

# load data
input_path = "/Users/jakewen/Desktop/Github/KOL_model/INPUT/central_bank_speech/all_speeches.xlsx"

speeches_data = pd.read_excel(input_path)

def str2array(s):
    # Remove space after [
    s=re.sub('\[ +', '[', s.strip())
    # Replace commas and spaces
    s=re.sub('[,\s]+', ', ', s)
    return np.array(ast.literal_eval(s))

speeches_data['text_embedding'] = speeches_data['text_embedding'].apply(str2array)

df_sample = speeches_data
df_sample.set_index('date', inplace=True)


#%% Embedding model
from sentence_transformers import SentenceTransformer

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
search_word = 'recession'

#%% Main body
print("program running")

Search_word_embedding = embedding(search_word)

df_keywords = df_sample.copy()

df_keywords[search_word] = df_keywords["text_embedding"].apply(lambda x: cosine_similarity_function(x, Search_word_embedding))

print("program completed")

#%% Post processing

date_range = pd.date_range(start=df_keywords.index.min(), end=df_keywords.index.max())
date_range = date_range.to_frame()
df_output = pd.merge(df_keywords, date_range, left_index=True, right_index=True, how='outer')
df_output.fillna(0,inplace=True)
df_output["value"] = df_output[search_word].rolling(window=180, min_periods=1).sum()

#%% Plot
import matplotlib.pyplot as plt

x = df_output.index
y = df_output["value"]

plt.xlabel("Date")
plt.ylabel(search_word+" value")
plt.title(search_word+" trend")

plt.plot(x, y)
plt.show()
