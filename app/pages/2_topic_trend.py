#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 13:46:53 2023

@author: jiayue.yuan
"""



#%% streamlit
import streamlit as st
#from functionforDownloadButtons import download_button

# data processing
import pandas as pd
import numpy as np
import datetime as dt
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import re
import ast
from numpy.linalg import norm

# plot
import matplotlib.pyplot as plt
import plotly.express as px
#pio.renderers.default = 'browser'

# embedding
from sentence_transformers import SentenceTransformer


st.set_page_config(
    page_title="Topic Trend Tracker",
    page_icon="🎈",
    layout="wide",
)

#%% Embedding Functions

def embedding(text):
    embedder = SentenceTransformer(embedder_name)
    doc_embeddings = embedder.encode(text)

    return doc_embeddings

# cosine similarity
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
    
    # plt method
    
    # plt.rcParams["figure.figsize"] = (10,6)
    
    # x = df_merged.index
    # y = df_merged[search_word+" value"]
    
    # plt.xlabel("Date")
    # plt.ylabel(search_word+" value")
    # plt.title(search_word)
    
    # fig = plt.plot(x, y)
    
    #ployly method

    x = df_merged.index
    y = df_merged[search_word+" value"]
    fig = px.line(df_merged, x, y, title=search_word_group+" trend")   
           
    # apply polarity
    df_merged = df_merged * polarity
    
    # apply topic group
    df_merged.columns = [search_word_group, search_word_group+" value"]
    
    return df_merged, fig

#%% plot summary chart

def plot_summary():
    

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

# x = df_output.index
# y = df_output["value"]
# fig = px.line(
#     df_output, 
#     x=df_output.index, 
#     y = df_output["value"],
#     title=search_word+" trend")
# fig.show()


#%% Export
#df_output.to_excel("df_output.xlsx")

#%% Treemap
def plot_treemap(df_output,eval_date,period_start, period_end):

    df = df_output.iloc[:,1::2]
    df.columns = df_output.iloc[:,::2].columns
    
    #convert df.index to dt.date
    df.index = pd.to_datetime(df.index)
    
    # get today's value
    df_today = df.loc[[eval_date]].T
    #df_today = df.sort_index().tail(1).T
    df_today = df_today.abs()
    
    # adjusted index (10 years from 2012-01-01)
    period_start = dt.date(2012,1,1)
    period_start = dt.date(2022,1,1)
    
    df_selected = df.loc[df.index >= str(period_start)]
    df_selected = df.loc[df.index <= str(period_end)]

    
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

#%% load data 

# Read in data from local.
@st.cache_data
def load_speeches_data(embedder_name,tag):

    #1.1 speech data
    #need to change to url(cloud adress)
    input_path = "/Users/yjy/Documents/GitHub/KOL_model/INPUT/central_bank_speech/" + embedder_name + "_embedding_" + tag + ".xlsx"
    speeches_data = pd.read_excel(input_path)
    
    return speeches_data
    
@st.cache_data
def load_reference_data():
    
    #1.2 reference data
    dic_reference_data = {}
    input_path_ref = "/Users/yjy/Documents/Github/KOL_model/INPUT/reference_tables/weight.xlsx"
    dic_reference_data["country_weight"] = pd.read_excel(input_path_ref, sheet_name="country")
    dic_reference_data["topic_list"] = pd.read_excel(input_path_ref, sheet_name="topic list")
    
    return dic_reference_data

# convert embedding string to array
def str2array(s):
    # Remove space after [
    s=re.sub('\[ +', '[', s.strip())
    # Replace commas and spaces
    s=re.sub('[,\s]+', ', ', s)
    return np.array(ast.literal_eval(s))


#%%pre-defined variables

li_embedder_names = ["all-MiniLM-L6-v2",'all-mpnet-base-v2']
li_tags = ["full","shortened"]


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

#%% main page

st.title(" 🎈 Topic Trend and Treemap")
st.header("")

with st.expander("ℹ️ - About this page", expanded=True):

    st.write(
        """     
-   We collect a list of meaningful topics. This page shows you the trends of each topic and a treemap. 
-   In the sidebar, you can flexibly adjust topic search and treemap setting.
	    """
    )

    st.markdown("")

st.markdown("")

st.markdown("### 🟠 Plase Choose an embedding model first: ")


#with st.form(key="my_form1"):
    
c1,c2 = st.columns([2,2])

with c1:
    embedder_name = st.selectbox("Select embedder",
                                 li_embedder_names)
    
with c2:      
    tag = st.selectbox("Select full or shortened text", li_tags)
    
#button1 = st.form_submit_button(label="✨ Load data!")
        
    #if not submit_button1:
        #st.stop()

#if st.button1('✨ Load data!'):
            
with st.spinner(text='Loading data'):

    speeches_data = load_speeches_data(embedder_name,tag)
    
    dic_reference_data  = load_reference_data()
    
    # data preprocessing
    reference_table_country = dic_reference_data["country_weight"]
    reference_table_topic_list = dic_reference_data["topic_list"]
    
    speeches_data = pd.merge(speeches_data, reference_table_country, on="country", how="inner")
    speeches_data['text_embedding'] = speeches_data['text_embedding'].apply(str2array)
    speeches_data.set_index('date', inplace=True)
    
    df_search_word = speeches_data.copy()
    
    # re-create date series
    date_range = pd.date_range(start=df_search_word.index.min(), end=df_search_word.index.max())
    date_range = date_range.to_frame()

    st.success('Data loaded suceddfully :sunglasses: ')

# add session state

#st.markdown("###### :green[_data loaded sucessfully_ :sunglasses:]")

#%% 3. sidebar for adjustable parameters

st.markdown("### 🟠 Adjust topic search and treemap settings in sidebar: ")

with st.sidebar:
    
    with st.form(key="my_form2"):
    
        st.write("🔸Topic search setting")
    
        # threshold levelcolorscales
        min_threshold = st.slider(
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
        power = st.number_input(
                "Power",
                value=6,
                min_value=1,
                max_value=10,
                help=""" The power when scale the similarity.
                The higher the setting, the greater the gap between high/low similarities will be magnified""",
            )
    
        
        st.write("🔸Treemap setting: ") 
        #evaluation date
        eval_date = st.date_input("🗓Choose evaluation date",
                                  value = dt.date(2022,11,10), # can change to Today() after go-live
                                  min_value= dt.date(2000,11,10), 
                                  max_value= dt.date(2022,11,10),
                                  help=""" To evaluate the market narratives on a specific date.
                                  """,)
            
        # Caution : parameters conflicts(eval_date & period), min date in dataframe : 1990-11-28
        period_start = st.date_input("🗓Choose period start date",
                                  value = dt.date(2012,11,10), # can change to Today() after go-live
                                  min_value= dt.date(2000,11,10), 
                                  max_value= dt.date(2022,11,10),
                                  help=""" The start date of period.
                                  """,)
                                  
        period_end = st.date_input("🗓Choose period end date",
                                  value = dt.date(2022,11,10), # can change to Today() after go-live
                                  min_value= dt.date(2000,11,10), 
                                  max_value= dt.date(2022,11,10),
                                  help=""" The end date of period.
                                  """,)
                                  
        submit_button2 = st.form_submit_button(label="✨ Get me the result!")
        
        if not submit_button2:
            st.stop() 
 
        if period_start > period_end:
            st.warning(" period end date can't be earlier than period start date")
            st.stop()

#%% 4. main loop for topic searching

placeholder = st.empty()
placeholder.text("Calculating......")
#st.write("Caculating......")

df_output = pd.DataFrame()
dic_figs = {}

for i in range(len(reference_table_topic_list)):
    
    search_word = reference_table_topic_list["child topics questions"][i]
    search_word_group = reference_table_topic_list["child topics"][i]
    polarity = reference_table_topic_list["polarity"][i]
    df_merged, fig = main_loop(search_word, search_word_group, polarity, df_search_word, date_range) 
    df_output = pd.concat([df_output, df_merged], axis=1)
    dic_figs[search_word_group] = fig

df_output = df_output.groupby(level=0, axis=1).sum()


fig_treemap = plot_treemap(df_output,eval_date,period_start,period_end)

placeholder.empty()

#%% main page layout

#compare two plots
st.markdown("##### 📈 View topic trends ")

ce, c1, ce, c2, ce = st.columns([0.2, 2, 0.2, 2, 0.2,])
with c1:
    topic1 = st.selectbox("select a topic",
                          list(dic_figs.keys()),
                          index = 0)
    st.plotly_chart(dic_figs[topic1]) 
    
    
with c2:  
    
    topic2 = st.selectbox("select a topic",
                          list(dic_figs.keys()),
                          index = 1)
    
    st.plotly_chart(dic_figs[topic2]) 
    
        
# summary_plot_checkbox = st.checkbox(
#     "🔸Display all topics trends",
#     help="Tick this box to diaplay all topic trends",
#         )
   
# if summary_plot_checkbox:
#     st.container()
#     st.pyplot()

st.markdown("### 📊 View topic Treemap ")
    
fig_treemap.update_layout(width = 1000, height=800)
st.plotly_chart(fig_treemap,width = 1000, height=800)

    























