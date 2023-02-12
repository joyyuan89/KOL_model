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
speeches_data.set_index('date', inplace=True)


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
search_word = 'crisis'
effective_date_list = [
    [30,1],     # within next 30 day, full impact
    [90,0.25],   # within next 60 day, half impact
    [180,0.1],  # within next 180 day, 1/10 impact
    ]
min_threshold = 0.15
power = 6

#%% Main body
Search_word_embedding = embedding(search_word)
df_keywords = speeches_data.copy()
df_keywords[search_word] = df_keywords["text_embedding"].apply(lambda x: cosine_similarity_function(x, Search_word_embedding))

df_keywords_copy = df_keywords.copy()

#%% Post processing
# adjust similarity value
def adjust_value(value):
    if value < min_threshold:
        value = 0
    value = value**power
    
    return value

df_keywords[search_word] = df_keywords[search_word].apply(lambda x: adjust_value(x))

# fill the date gaps
date_range = pd.date_range(start=df_keywords.index.min(), end=df_keywords.index.max())
date_range = date_range.to_frame()
df_output = pd.merge(df_keywords, date_range, left_index=True, right_index=True, how='outer')

# create value with time decay
df_output["value"] = 0
for i in effective_date_list:
    df_output["value"] += df_output[search_word].rolling(window=i[0], min_periods=1).sum()*i[1]

#%% Plot
import matplotlib.pyplot as plt

x = df_output.index
y = df_output["value"]

plt.xlabel("Date")
plt.ylabel(search_word+" value")
plt.title(search_word+" trend")

plt.plot(x, y)
plt.show()

#%% Plotly
import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'
#pio.renderers.default = 'svg'

x = df_output.index
y = df_output["value"]

fig = px.line(
    df_output, 
    x=df_output.index, 
    y = df_output["value"],
    title=search_word+" trend")

fig.show()

#%% Export

df_output.to_excel("df_output.xlsx")