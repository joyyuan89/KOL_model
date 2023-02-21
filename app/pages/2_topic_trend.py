#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 13:46:53 2023

@author: jiayue.yuan
"""

#%%
# libraries
import pandas as pd
import numpy as np
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import re
import ast
import matplotlib.pyplot as plt

#%% Embedding Functions

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


#%% topic searching functions

# adjust similarity value
def adjust_value(value):
    if value < min_threshold:
        value = 0
    value = value**power
    
    return value

# main loop
def main_loop(search_word, search_word_group, polarity, df_search_word, date_range, individual_plot):

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
    dic_figs = {}
    
    if individual_plot:    
        
        plt.rcParams["figure.figsize"] = (10,6)
        
        x = df_merged.index
        y = df_merged[search_word+" value"]
        
        plt.xlabel("Date")
        plt.ylabel(search_word+" value")
        plt.title(search_word)
        
        fig = plt.plot(x, y)
        dic_figs[search_word] = fig
        #plt.show()
    
    # apply polarity
    df_merged = df_merged * polarity
    
    # apply topic group
    df_merged.columns = [search_word_group, search_word_group+" value"]
    
    return df_merged, dic_figs

#%% plot summary chart

def plot_summary():
    
    summary_plot

    if len(df_output.columns) > 2:
        n_col = 2
        width = n_col
        height = np.ceil(len(df_output.columns)/n_col).astype(int)
        plt.rcParams["figure.figsize"] = (width*10,height*5)
        fig, ax = plt.subplots(nrows=height, ncols=width)
        
        for i in range(len(df_output.columns)):
            ax[int(i/n_col),i%n_col].plot(df_output.iloc[:,i])
            ax[int(i/n_col),i%n_col].set_title(df_output.columns[i])

        
    return fig

        #plt.show()

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
#df_output.to_excel("df_output.xlsx")

#%% Treemap

import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'

def plot_treemap(df_output,eval_date,period_start, period_end):

    df = df_output.iloc[:,1::2]
    df.columns = df_output.iloc[:,::2].columns
    
    # get today's value
    df_today = df.loc[[eval_date]].T
    #df_today = df.sort_index().tail(1).T
    df_today = df_today.abs()
    
    # adjusted index (10 years from 2012-01-01)
    df_selected = df.loc[df.index >= period_start & df.index <= period_end]
    
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
 
    
#fig = plot_treemap(df_output,"2022-11-10")

#fig.show()



#%% streamlit app
import streamlit as st
from functionforDownloadButtons import download_button

st.set_page_config(
    page_title="Topic Trend Tracker",
    page_icon="🎈",
    layout="wide",
)

#%% 1. load data 

# Read in data from local.
@st.cache
def load_data(li_embedder_names,li_tags):

    #1.1 speech data
    dic_speeches_data = {}
    #need to change to url(cloud adress)
    li_input_paths = ["/Users/jiayue.yuan/Documents/GitHub/KOL_model/INPUT/central_bank_speech/all-MiniLM-L6-v2_embedding_full.xlsx",
                      "/Users/jiayue.yuan/Documents/GitHub/KOL_model/INPUT/central_bank_speech/all-MiniLM-L6-v2_embedding_shortened.xlsx",
                      "/Users/jiayue.yuan/Documents/GitHub/KOL_model/INPUT/central_bank_speech/all-mpnet-base-v2_embedding_full.xlsx"]
    
    dic_speeches_data[li_embedder_names[0]] = {}
    dic_speeches_data[li_embedder_names[0]][li_tags[0]] = pd.read_excel(li_input_paths[0])
    dic_speeches_data[li_embedder_names[0]][li_tags[1]] = pd.read_excel(li_input_paths[1])
    dic_speeches_data[li_embedder_names[1]] = {li_tags[0]:pd.read_excel(li_input_paths[2])}
    
    #1.2 reference data
    dic_reference_data = {}
    input_path_ref = "/Users/jiayue.yuan/Documents/Github/KOL_model/INPUT/reference_tables/weight.xlsx"
    dic_reference_data["country_weight"] = pd.read_excel(input_path_ref, sheet_name="country")
    dic_reference_data["topic_list"] = pd.read_excel(input_path_ref, sheet_name="topic list")
    
    return dic_speeches_data, dic_reference_data

# convert embedding string to array
def str2array(s):
    # Remove space after [
    s=re.sub('\[ +', '[', s.strip())
    # Replace commas and spaces
    s=re.sub('[,\s]+', ', ', s)
    return np.array(ast.literal_eval(s))

li_embedder_names = ["all-MiniLM-L6-v2",'all-mpnet-base-v2']
li_tags = ["full","shortened"]
dic_speeches_data, dic_reference_data = load_data(li_embedder_names,li_tags)

#%% 2. sidebar for embedding data

embedder_name = st.sidebar.selectbox("Select embedder", li_embedder_names)
tag = st.sidebar.selectbox("Select full or shortened text", li_tags)

# choose topic_list(to add more lists later)
reference_table_country = dic_reference_data["country_weight"]
reference_table_topic_list = dic_reference_data["topic_list"]

# select data and preprocessing
speeches_data = dic_speeches_data[embedder_name][tag]
speeches_data = dic_speeches_data(speeches_data, reference_table_country, on="country", how="inner")
speeches_data['text_embedding'] = dic_speeches_data['text_embedding'].apply(str2array)
speeches_data.set_index('date', inplace=True)

df_search_word = speeches_data.copy()

# re-create date series
date_range = pd.date_range(start=df_search_word.index.min(), end=df_search_word.index.max())
date_range = date_range.to_frame()

#%% 3. sidebar for adjustable parameters

# time decay (to replace with functions later)
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

individual_plot = True
summary_plot = True

with st.sidebar():
    
    with st.form(key="my_form"):
    
        st.write("Topic search setting")
    
        # threshold levelcolorscales
        min_threshold = st.sidebar.slider(
                "Min threshold",
                value=0.1,
                min_value=0.1,
                max_value=0.9,
                step=0.1,
                help=""" The min threshold of similarity between a speech and a topic when searching related speeches.
                The higher the setting, the more speeches will be found related to specified topic.
                """,
            )
        
        # scaling factor
        power = st.sidebar.number_input(
                "Power",
                value=6,
                min_value=1,
                max_value=10,
                help=""" The power when scale the similarity.
                The higher the setting, the greater the gap between high/low similarities will be magnified""",
            )
    
        
        st.write("Plot setting: ") 
        
        individual_plot_checkbox = st.checkbox(
            "plot trend of each topic",
            help="Tick this box to plot trend of each topic",
        )
        
        summary_plot_checkbox = st.checkbox(
            "plot trends of topics together",
            help="Tick this box to plot trends of topics",
        )
        
        st.write("Treemap setting: ") 
        #evaluation date
        eval_date = st.date_input("🗓Choose evaluation date",
                                  value = "2022-11-10", # can change to Today() after go-live
                                  min_value= "2000-11-10", 
                                  max_value= "2022-11-10",
                                  help=""" To evaluate the market narratives on a specific date.
                                  """,)
            
        # Caution : parameters conflicts(eval_date & period), min date in dataframe : 1990-11-28
        period_start = st.date_input("🗓Choose evaluation date",
                                  value = "2012-11-10", # can change to Today() after go-live
                                  min_value= "2000-11-10", 
                                  max_value= "2022-11-10",
                                  help=""" The start date of period.
                                  """,)
                                  
        period_end = st.date_input("🗓Choose evaluation date",
                                  value = "2022-11-10", # can change to Today() after go-live
                                  min_value= "2000-11-10", 
                                  max_value= "2022-11-10",
                                  help=""" The end date of period.
                                  """,)
                                  
        submit_button = st.form_submit_button(label="✨ Get me the result!")

        if individual_plot_checkbox:
            if_individual_plot = True
            
        if summary_plot_checkbox:
            if_summary_plot = True
 
if not submit_button:
    st.stop()   
 
if period_start > period_end:
    st.warning(" period end date can't be earlier than period start date")
    st.stop()

#%% 4. main loop for topic searching

df_output = pd.DataFrame()

for i in range(len(reference_table_topic_list)):
    search_word = reference_table_topic_list["child topics questions"][i]
    search_word_group = reference_table_topic_list["child topics"][i]
    polarity = reference_table_topic_list["polarity"][i]
    df_merged, dic_figs = main_loop(search_word, search_word_group, polarity, df_search_word, date_range) # figs
    df_output = pd.concat([df_output, df_merged], axis=1)

df_output = df_output.groupby(level=0, axis=1).sum()


if summary_plot:
    fig_summary = plot_summary()

fig_treemap = plot_treemap(df_output,eval_date)

#%% main page layout

st.title(" 🎈 Topic Trend and Treemap")
st.header("")

#compare two plots
st.markdown("### 📈 View topic trends ")

ce, c1, ce, c2, ce = st.columns([0.2, 2, 0.2, 2, 0.2,])
with c1:
    topic1 = st.selectbox("select a topic",
                          list(dic_figs.keys()),
                          index = 0)
    
    st.pyplot(dic_figs[topic1]) 
    
    
with c2:  
    
    topic2 = st.selectbox("select a topic",
                          list(dic_figs.keys()),
                          index = 1)
    
    st.pyplot(dic_figs[topic2]) 

st.markdown("#### 📈Check & download results")

st.header("")

cs, c1, c2, cLast = st.columns([2, 1.5, 1.5, 2])

with c1:
    CSVButton2 = download_button(df_output, "Topic_Trend.csv", "📥 Download Topic Trends data (.csv)")
with c2:
    st.write("placeholder for jpeg download")
    #CSVButton2 = download_button(df_output, "Trend_pic.jpeg", "📥 Download (.jpeg)")
    
if summary_plot:
    st.pyplot(fig_summary) 


st.markdown("### 📊 View topic Treemap ")
st.plotly_chart(fig_treemap)
    

# download summary plot






















