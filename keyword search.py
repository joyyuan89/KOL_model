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
import matplotlib.pyplot as plt

# dir
work_dir = os.getcwd()

# embedder name
# embedder_name = 'multi-qa-MiniLM-L6-cos-v1'
embedder_name = 'all-MiniLM-L6-v2'

# load data
input_path = "/Users/jakewen/Desktop/Github/KOL_model/INPUT/central_bank_speech/"+embedder_name+"_embedding.xlsx"
speeches_data = pd.read_excel(input_path)

# convert embedding string to array
def str2array(s):
    # Remove space after [
    s=re.sub('\[ +', '[', s.strip())
    # Replace commas and spaces
    s=re.sub('[,\s]+', ', ', s)
    return np.array(ast.literal_eval(s))

speeches_data['text_embedding'] = speeches_data['text_embedding'].apply(str2array)
speeches_data.set_index('date', inplace=True)

df_search_word = speeches_data.copy()

# re-create date series
date_range = pd.date_range(start=df_search_word.index.min(), end=df_search_word.index.max())
date_range = date_range.to_frame()


#%% Embedding model
from sentence_transformers import SentenceTransformer

def embedding(text):
    embedder = SentenceTransformer(embedder_name)
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
    'raise interest rate',
    'commodity price',
    'crypto bitcoin',
    'recession',
    'mortgage affortability',
    'quantitative easing',
    'geopolitical conflict',
    'great depression',
    'stagflation',
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
def main_loop(search_word, df_search_word, date_range):

    # search word embedding
    search_word_embedding = embedding(search_word)

    # calculate cosine similarity between search word and articles
    df_search_word[search_word] = df_search_word["text_embedding"].apply(lambda x: cosine_similarity_function(x, search_word_embedding))
    df_search_word[search_word] = df_search_word[search_word].apply(lambda x: adjust_value(x))
    
    # re-index to daily frequency and sum the values
    df_merged = pd.merge(
        date_range, 
        df_search_word, 
        how='left',
        left_index=True, 
        right_index=True)

    df_merged = df_merged.resample('D')[search_word].sum()
    df_merged = df_merged.to_frame(search_word)
    
    # create value with time decay
    df_merged[search_word+" value"] = 0
    for i in effective_date_list:
        df_merged[search_word+" value"] += df_merged[search_word].rolling(window=i[0], min_periods=1).sum()*i[1]
    
    # display top n relevant news
    df_relevant = df_merged.loc[df_merged[search_word] != 0]
    df_relevant = df_relevant.sort_values(by=[search_word], ascending=False).head(5)
    
    print(df_relevant)
    
    # plot
    plt.rcParams["figure.figsize"] = (10,6)
    
    x = df_merged.index
    y = df_merged[search_word+" value"]
    
    plt.xlabel("Date")
    plt.ylabel(search_word+" value")
    plt.title(search_word)
    
    plt.plot(x, y)
    plt.show()
    
    return df_merged

df_output = pd.DataFrame()

for search_word in search_word_list:
    df_merged = main_loop(search_word, df_search_word, date_range)
    df_output = pd.concat([df_output, df_merged], axis=1)

# plot summary chart
if len(search_word_list) > 2:
    n_col = 2
    width = n_col
    height = np.ceil(len(search_word_list)/n_col).astype(int)
    plt.rcParams["figure.figsize"] = (width*10,height*5)
    fig, ax = plt.subplots(nrows=height, ncols=width)
    
    count = 0
    for search_word in search_word_list:
        ax[int(count/n_col), count%n_col].plot(df_output.iloc[:, count*2+1])
        ax[int(count/n_col), count%n_col].set_title(search_word)
        count += 1
    
    plt.show()
else:
    pass

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
df_output.to_excel("df_output.xlsx")
'''