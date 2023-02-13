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
df_keywords = speeches_data.copy()

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

# search word list
search_word_list = [
    'increasing inflation',
    'banking crisis',
    'economy recession',
    'crypto bitcoin',
    'rescue economy',
    'covid',
    'oil price',
    'geopolitical conflict',
    ]

# time decay
effective_date_list = [
    [15,0.10],
    [30,0.10],
    [45,0.10],
    [60,0.10],
    [90,0.10],
    [120,0.10],
    [150,0.10],
    [180,0.10],
    [270,0.10],
    [360,0.10],
    ]

# threshold level
min_threshold = 0.10

# scaling factor
power = 6

#%% Main body

# adjust similarity value
def adjust_value(value):
    if value < min_threshold:
        value = 0
    value = value**power
    
    return value

# main loop
def main_loop(search_word, df_keywords):

    search_word_embedding = embedding(search_word)

    df_keywords[search_word] = df_keywords["text_embedding"].apply(lambda x: cosine_similarity_function(x, search_word_embedding))
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
    
    return df_output
    
for search_word in search_word_list:
    df_output = main_loop(search_word, df_keywords)
    

#%% Plotly
'''
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
'''