#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 13:46:53 2023

@author: jiayue.yuan
"""


#import streamlit as st

# libraries
import pandas as pd
import numpy as np
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import re
import ast
import matplotlib.pyplot as plt


#%%
# 1.load data

dic_data = {}
li_embedder_names = ["all-MiniLM-L6-v2",'all-mpnet-base-v2']
li_tags = ["full","shortened"]
#need to change to url(cloud adress)
li_input_paths = ["/Users/jiayue.yuan/Documents/GitHub/KOL_model/INPUT/central_bank_speech/all-MiniLM-L6-v2_embedding_full.xlsx",
                  "/Users/jiayue.yuan/Documents/GitHub/KOL_model/INPUT/central_bank_speech/all-MiniLM-L6-v2_embedding_shortened.xlsx",
                  "/Users/jiayue.yuan/Documents/GitHub/KOL_model/INPUT/central_bank_speech/all-mpnet-base-v2_embedding_full.xlsx"]

dic_data[li_embedder_names[0]] = {}
dic_data[li_embedder_names[0]][li_tags[0]] = pd.read_excel(li_input_paths[0])
dic_data[li_embedder_names[0]][li_tags[1]] = pd.read_excel(li_input_paths[1])
dic_data[li_embedder_names[1]] = {li_tags[0]:pd.read_excel(li_input_paths[2])}

input_path_ref = "/Users/jiayue.yuan/Documents/Github/KOL_model/INPUT/reference_tables/weight.xlsx"
reference_table_country = pd.read_excel(input_path_ref, sheet_name="country")
reference_table_topic_list = pd.read_excel(input_path_ref, sheet_name="topic list")

# convert embedding string to array
def str2array(s):
    # Remove space after [
    s=re.sub('\[ +', '[', s.strip())
    # Replace commas and spaces
    s=re.sub('[,\s]+', ', ', s)
    return np.array(ast.literal_eval(s))

#%% Function list

# embedding model
from sentence_transformers import SentenceTransformer

def embedding(text):
    embedder = SentenceTransformer(embedder_name)
    doc_embeddings = embedder.encode(text)

    return doc_embeddings

# cosine similarity
from numpy.linalg import norm

def cosine_similarity_function(vec_1, vec_2):
    value = np.dot(vec_1, vec_2.T)/(norm(vec_1)*norm(vec_2))
    return value

#%% Variables
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

# threshold levelcolorscales
min_threshold = 0.10

# scaling factor
power = 6

individual_plot = True
summary_plot = True


# embedder and tag
embedder_name = li_embedder_names[0]
tag = li_tags[0]

speeches_data = dic_data[embedder_name][tag]
speeches_data = pd.merge(speeches_data, reference_table_country, on="country", how="inner")
speeches_data['text_embedding'] = speeches_data['text_embedding'].apply(str2array)
speeches_data.set_index('date', inplace=True)

df_search_word = speeches_data.copy()

# re-create date series
date_range = pd.date_range(start=df_search_word.index.min(), end=df_search_word.index.max())
date_range = date_range.to_frame()

#%% Main body

# adjust similarity value
def adjust_value(value):
    if value < min_threshold:
        value = 0
    value = value**power
    
    return value

# main loop
def main_loop(search_word, search_word_group, polarity, df_search_word, date_range):

    # search word embedding
    search_word_embedding = embedding(search_word)

    # calculate cosine similarity between search word and articles
    df_search_word[search_word] = df_search_word["text_embedding"].apply(lambda x: cosine_similarity_function(x, search_word_embedding))
    df_search_word[search_word] = df_search_word[search_word].apply(lambda x: adjust_value(x))
    df_search_word[search_word] = df_search_word[search_word]*df_search_word["country_weight"]
    
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
    
    #print(df_relevant)

    # plot individual plot
    if individual_plot:    
        
        plt.rcParams["figure.figsize"] = (10,6)
        
        x = df_merged.index
        y = df_merged[search_word+" value"]
        
        plt.xlabel("Date")
        plt.ylabel(search_word+" value")
        plt.title(search_word)
        
        plt.plot(x, y)
        plt.show()
    
    # apply polarity
    df_merged = df_merged * polarity
    
    # apply topic group
    df_merged.columns = [search_word_group, search_word_group+" value"]
    
    return df_merged

df_output = pd.DataFrame()

for i in range(len(reference_table_topic_list)):
    search_word = reference_table_topic_list["child topics questions"][i]
    search_word_group = reference_table_topic_list["child topics"][i]
    polarity = reference_table_topic_list["polarity"][i]
    df_merged = main_loop(search_word, search_word_group, polarity, df_search_word, date_range)
    df_output = pd.concat([df_output, df_merged], axis=1)

df_output = df_output.groupby(level=0, axis=1).sum()
#%% plot summary chart

if summary_plot:

    if len(df_output.columns) > 2:
        n_col = 2
        width = n_col
        height = np.ceil(len(df_output.columns)/n_col).astype(int)
        plt.rcParams["figure.figsize"] = (width*10,height*5)
        fig, ax = plt.subplots(nrows=height, ncols=width)
        
        for i in range(len(df_output.columns)):
            ax[int(i/n_col),i%n_col].plot(df_output.iloc[:,i])
            ax[int(i/n_col),i%n_col].set_title(df_output.columns[i])

        plt.show()
    else:
        pass

#%% Plotly
    
'''    
import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'
x = df_output.index
y = df_output["value"]
fig = px.line(
    df_output, 
    x=df_output.index, 
    y = df_output["value"],
    title=search_word+" trend")
fig.show()
'''

#%% Export
df_output.to_excel("df_output.xlsx")

#%% Treemap

import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'

def plot_treemap(df_output,eval_date):

    df = df_output.iloc[:,1::2]
    df.columns = df_output.iloc[:,::2].columns
    
    # get today's value
    df_today = df.loc[[eval_date]].T
    #df_today = df.sort_index().tail(1).T
    df_today = df_today.abs()
    
    # adjusted index (10 years from 2012-01-01)
    df_selected = df.loc[df.index >= "2012-01-01"]
    
    df_adjusted = ((df_selected -df_selected.min())/(df_selected.max() - df_selected.min()))
    df_adjusted_today = df_adjusted.loc[[eval_date]].T
    df_result = pd.concat([df_today,df_adjusted_today],axis = 1)
    df_result.reset_index(inplace = True)
    df_result.columns = ["child topics", "value","adj_value"]
    
    parent_topics = reference_table_topic_list.drop_duplicates(subset="child topics")
    df_result_final = pd.merge(df_result, 
                          parent_topics, 
                          on ='child topics', 
                          how ='inner')
    
    # create a treemap of the data using Plotly
    fig = px.treemap(df_result_final, 
                     path=[px.Constant('Market topics'), 'parent topics', 'child topics'],
                     values='value',
                     color='adj_value', 
                     #color_continuous_scale='RdBu_r',
                      color_continuous_scale='oranges',
                     hover_data={'value':':.2f', 'adj_value':':d'})
    
    # show the treemap
    #fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
    fig.update_layout(font_size=20,font_family="Open Sans",font_color="#444")
    
    return fig
 
    
fig = plot_treemap(df_output,"2022-11-10")

fig.show()