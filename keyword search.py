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
# embedder_name = "all-MiniLM-L6-v2"
embedder_name = 'all-mpnet-base-v2' # heavy weight all-rounder

# full or shortened text
tag = "full"

# load data
input_path_data = "/Users/jakewen/Desktop/Github/KOL_model/INPUT/central_bank_speech/"+embedder_name+"_embedding_"+tag+".xlsx"
speeches_data = pd.read_excel(input_path_data)
input_path_ref = "/Users/jakewen/Desktop/Github/KOL_model/INPUT/reference_tables/weight.xlsx"
reference_table_country = pd.read_excel(input_path_ref, sheet_name="country")

speeches_data = pd.merge(speeches_data, reference_table_country, on="country", how="inner")

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

# search word list

reference_table_topic_list = pd.read_excel(input_path_ref, sheet_name="topic list")

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

# scaling factor (default is 4)
power = 4

individual_plot = False
summary_plot = True

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

# main looop
print("main loop running...")
for i in range(len(reference_table_topic_list)):
    search_word = reference_table_topic_list["child topics questions"][i]
    search_word_group = reference_table_topic_list["child topics"][i]
    polarity = reference_table_topic_list["polarity"][i]
    df_merged = main_loop(search_word, search_word_group, polarity, df_search_word, date_range)
    df_output = pd.concat([df_output, df_merged], axis=1)
print("main loop completed")

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

df = df_output.iloc[:,1::2]
df.columns = df_output.iloc[:,::2].columns

# get today's value
df_today = df.sort_index().tail(1).unstack()
df_today.reset_index(level=-1, drop=True, inplace=True)
df_today = df_today.abs()

# adjusted index with some time decay (testing)
selection_list = [
    ['1990-01-01', '2008-01-01', 5],
    ['2008-01-01', '2018-01-01', 2],
    ['2018-01-01', '2023-01-01', 1],
    ]

df_selected = pd.DataFrame()
for item in selection_list:
    df_temp = df.loc[(df.index >= item[0])&(df.index < item[1])]
    df_temp = df_temp.iloc[::item[2], :]
    df_selected = pd.concat([df_selected, df_temp], axis=0)


# 2 methods of calculating adjusted value

# 1st method is to calculate relative percentage of min and max
# df_adjusted = ((df_selected -df_selected.min())/(df_selected.max() - df_selected.min())).tail(1).T

# 2nd method is to calculate the percentile of the last value
df_adjusted = df_selected.rank(pct=True).tail(1).T

df_result = pd.concat([df_today,df_adjusted],axis = 1)
df_result.reset_index(inplace = True)
df_result.columns = ['child topics', 'popularity', 'severity']

parent_topics = reference_table_topic_list.drop_duplicates(subset='child topics')
df_result_final = pd.merge(df_result, 
                      parent_topics, 
                      on ='child topics', 
                      how ='inner')


# plot treemap
import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'

# create a treemap of the data using Plotly
fig = px.treemap(df_result_final, 
                 path=[px.Constant('Market topics'), 'parent topics', 'child topics'],
                 values='popularity',
                 color='severity', 
                 #color_continuous_scale='RdBu_r',
                  color_continuous_scale='oranges',
                 hover_data={'popularity':':.2f', 'severity':':.2f'})

# show the treemap
#fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))

fig.update_layout(font_size=20,font_family="sans-serif ",font_color="#444")
fig.show()