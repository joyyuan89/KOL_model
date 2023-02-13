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
os.environ["TOKENIZERS_PARALLELISM"] = "false"
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
search_word = 'rescue economy'
effective_date_list = [
    [15,0.5],       # within next 15 day, full impact
    [30,0.25],      # within next 30 day, half impact
    [90,0.15],      # within next 90 day, quarter impact
    [180,0.1],      # within next 180 day, 1/10 impact
    ]
min_threshold = 0.10
power = 6

#%% Main body
search_word_embedding = embedding(search_word)
df_keywords = speeches_data.copy()
df_keywords[search_word] = df_keywords["text_embedding"].apply(lambda x: cosine_similarity_function(x, search_word_embedding))

df_keywords_copy = df_keywords.copy()

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
df_output.fillna(0, inplace=True)

# create value with time decay
df_output["value"] = 0
for i in effective_date_list:
    df_output["value"] += df_output[search_word].rolling(window=i[0], min_periods=1).sum()*i[1]

# display top n relevant news
df_relevant = df_output.loc[df_output[search_word] != 0]
df_relevant = df_relevant.sort_values(by=[search_word], ascending=False).head(10)

print(df_relevant)

# plot
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,6)

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
# df_output.to_excel("df_output.xlsx")