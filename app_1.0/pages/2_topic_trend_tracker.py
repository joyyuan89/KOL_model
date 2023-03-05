#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 13:46:53 2023

@author: jiayue.yuan
"""



#%% streamlit
import streamlit as st

# data processing
import pandas as pd
import numpy as np
import datetime as dt
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import re
import ast
from numpy.linalg import norm


from google.oauth2 import service_account
from google.cloud import storage

# plot
import plotly.express as px

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
def adjust_value(value,power,min_threshold):
    if value < min_threshold:
        value = 0
    value = value**power
    
    return value

# main loop
def main_loop(search_word, search_word_group, polarity, df_search_word, date_range, power,min_threshold):

    # search word embedding
    search_word_embedding = embedding(search_word)

    # calculate cosine similarity between search word and articles
    df_search_word[search_word] = df_search_word["text_embedding"].apply(lambda x: cosine_similarity_function(x, search_word_embedding))
    df_search_word[search_word] = df_search_word[search_word].apply(lambda x: adjust_value(x,power,min_threshold))
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
    
    #ployly method

    x = df_merged.index
    y = df_merged[search_word+" value"]
    fig = px.line(df_merged, x, y, title=search_word_group+" trend")   
           
    # apply polarity
    df_merged = df_merged * polarity
    
    # apply topic group
    df_merged.columns = [search_word_group, search_word_group+" value"]
    
    return df_merged, fig

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


@st.cache_data
def main_func(reference_table_topic_list,df_search_word, date_range, 
              eval_date,period_start,period_end,
              power,min_threshold):

    df_output = pd.DataFrame()
    dic_figs = {}
    
    for i in range(len(reference_table_topic_list)):
        
        search_word = reference_table_topic_list["child topics questions"][i]
        search_word_group = reference_table_topic_list["child topics"][i]
        polarity = reference_table_topic_list["polarity"][i]
        df_merged, fig = main_loop(search_word, search_word_group, polarity, df_search_word, date_range, power, min_threshold) 
        df_output = pd.concat([df_output, df_merged], axis=1)
        dic_figs[search_word_group] = fig
    
    df_output = df_output.groupby(level=0, axis=1).sum()
    
    fig_treemap = plot_treemap(df_output,eval_date,period_start,period_end)
    
    return dic_figs,fig_treemap

#%% load data 

# convert embedding string to array
def str2array(s):
    # Remove space after [
    s=re.sub('\[ +', '[', s.strip())
    # Replace commas and spaces
    s=re.sub('[,\s]+', ', ', s)
    return np.array(ast.literal_eval(s))


# Create API client.
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = storage.Client(credentials=credentials)

# Retrieve file contents.
def read_excel_file(bucket_name, file_path):
    bucket = client.bucket(bucket_name)
    excel_file_content = bucket.blob(file_path).download_as_bytes()
    
    return excel_file_content

@st.cache_data
def load_speeches_data(embedder_name,tag):

    bucket_name = "kol_model"
    file_path = "INPUT/central_bank_speech/" + embedder_name + "_embedding_" + tag + ".xlsx"
    file_content = read_excel_file(bucket_name, file_path)
    speeches_data = pd.read_excel(file_content)
    
    return speeches_data
    
@st.cache_data
def load_reference_data():

    bucket_name = "kol_model"
    file_path= "INPUT/reference_tables/weight.xlsx"
    file_content = read_excel_file(bucket_name, file_path)
    
    dic_reference_data = {}
    dic_reference_data["country_weight"] = pd.read_excel(file_content, sheet_name="country")
    dic_reference_data["topic_list"] = pd.read_excel(file_content, sheet_name="topic list")
    
    return dic_reference_data

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

st.title(" 📈 Topic Trend Tracker")
st.header("")

with st.expander("ℹ️ - About this page", expanded=True):

    st.write(
        """     
- We have collected a list of meaningful topics and created a tool to track their trends over time. 
- On this page, you can explore the trends of each topic and view a treemap of all topics.
	    """
    )

    st.markdown("")

st.markdown("")
#%% load data
st.markdown("### 🟠 1. Specify the embedder and speech length and load data")

c1,black,c2 = st.columns([2,1,2])

with c1:
    embedder_name = st.selectbox("Select embedder",li_embedder_names)
    
with c2:      
    tag = st.selectbox("Select full or shortened text", li_tags)
    
button_load_data = st.button(label="✨ Load data!", help="click the bottom when embedder and tag change")
# --- Initialising SessionState ---
if "load_state" not in st.session_state:
     st.session_state["load_state"] = False
     
if "data" not in st.session_state:
    st.session_state["data"] = {}

       
if button_load_data:
            
    with st.spinner(text='Loading data'):
        
        data = {}
    
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
        
        data["speeches_data"] = speeches_data
        data["reference_table_topic_list"] = reference_table_topic_list
        data["reference_table_country"] = reference_table_country
        data["df_search_word"] = df_search_word
        data["date_range"] = date_range
        
        st.session_state["load_state"] = True
        
        st.success('Data loaded successfully :sunglasses: ')
    
    st.session_state["data"] = data
    

#%% 3. sidebar for adjustable parameters

if st.session_state["load_state"]:
    
    st.markdown("### 🟠 2.Select the model parameters and calculate topic trends")
    
    st.markdown("🔹 We already pre-defined a topic list (see the last page :blue[**_More_**] to view the list.) \
                    Please adjust the topic search, trend index calculation and treemap settings in the sidebar, and click the :blue[**_Get me the result_**] botton to view results!")
    
    with st.sidebar:
        
        #with st.form(key="my_form2"):
        
            st.write("#### 🔸Topic search setting")
        
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
                    key="min_threhold"
                )
            
            # scaling factor
            power = st.number_input(
                    "Power",
                    value=6,
                    min_value=1,
                    max_value=10,
                    help=""" The power when scale the similarity.
                    The higher the setting, the greater the gap between high/low similarities will be magnified""",
                    key = "power"
                )
            
            st.write("#### 🔸Treemap setting: ") 
            #evaluation date
            eval_date = st.date_input("🗓Choose evaluation date",
                                      value = dt.date(2022,11,10), # can change to Today() after go-live
                                      min_value= dt.date(2000,11,10), 
                                      max_value= dt.date(2022,11,10),
                                      help=""" To evaluate the market narratives on a specific date.
                                      """,
                                      key = "eval_date")
                
            # Caution : parameters conflicts(eval_date & period), min date in dataframe : 1990-11-28
            period_start = st.date_input("🗓Choose period start date",
                                      value = dt.date(2012,11,10), # can change to Today() after go-live
                                      min_value= dt.date(2000,11,10), 
                                      max_value= dt.date(2022,11,10),
                                      help=""" The start date of period.
                                      """,
                                      key = "period_start")
                                      
            period_end = st.date_input("🗓Choose period end date",
                                      value = dt.date(2022,11,10), # can change to Today() after go-live
                                      min_value= dt.date(2000,11,10), 
                                      max_value= dt.date(2022,11,10),
                                      help=""" The end date of period.
                                      """,
                                      key = "period_end")
                                      
            #submit_button2 = st.form_submit_button(label="✨ Get me the result!")
            
            # if not submit_button2:
            #     st.stop() 
     
            # if period_start > period_end:
            #     st.warning(" period end date can't be earlier than period start date")
            #     st.stop()
            
            
            button_cal = st.button(label="✨ Get me the result!")
        # --- Initialising SessionState ---
        
    # if "cal_state" not in st.session_state:
    #     st.session_state["cal_state"] = False
    
    if "cal_state" not in st.session_state:
        st.session_state["cal_state"] = False
        
    if "dic_figs" not in st.session_state:
        st.session_state["dic_figs"] = {}
        
    if "fig_treemap" not in st.session_state:
        st.session_state["fig_treemap"] = {}
    

    if button_cal:
           
        data  = st.session_state["data"]
        
        speeches_data = data["speeches_data"]
        reference_table_country = data["reference_table_country"]
        reference_table_topic_list = data["reference_table_topic_list"]
        df_search_word = data["df_search_word"]
        date_range = data["date_range"] 
                
    
#%% 4. main loop for topic searching
            
        placeholder = st.empty()
        placeholder.text("Calculating......")
        #st.write("Caculating......")
        
        dic_figs,fig_treemap = main_func(reference_table_topic_list,df_search_word, date_range,
                                         eval_date,period_start,period_end,
                                         power,min_threshold)
        
        st.session_state["dic_figs"] = dic_figs
        st.session_state["fig_treemap"] = fig_treemap 
        
        st.session_state["cal_state"] = True
        
        placeholder.empty()
    
#%% view plots
    
if st.session_state["load_state"] and st.session_state["cal_state"]:
    st.markdown("##### 📈 View Topic Trends ")
    
    dic_figs = st.session_state["dic_figs"]
    fig_treemap = st.session_state["fig_treemap"]
    
    ce, c1, ce, c2, ce = st.columns([0.2, 2, 0.2, 2, 0.2,])
    
    with c1:
        topic1 = st.selectbox("select a topic",
                              list(dic_figs.keys()),
                              index = 0)
        with st.container():
            st.plotly_chart(dic_figs[topic1],use_container_width=True) 
        
        
    with c2:  
        
        topic2 = st.selectbox("select a topic",
                              list(dic_figs.keys()),
                              index = 1)
        
        with st.container():
            st.plotly_chart(dic_figs[topic2],use_container_width=True) 
        
    cf, c3, cf, c4, cf = st.columns([0.2, 2, 0.2, 2, 0.2,])
    
    with c3:
        topic3 = st.selectbox("select a topic",
                              list(dic_figs.keys()),
                              index = 2)
        
        with st.container():
            st.plotly_chart(dic_figs[topic3],use_container_width=True) 
        
       
    with c4:  
        
        topic4 = st.selectbox("select a topic",
                              list(dic_figs.keys()),
                              index = 3)
        
        with st.container():
            st.plotly_chart(dic_figs[topic4],use_container_width=True) 
           
    st.markdown("#### 📊 View Topic Treemap ")
        
    fig_treemap.update_layout(width = 1000, height=800)
    st.plotly_chart(fig_treemap,width = 1000, height=800)
    
    

    st.markdown("""##### 📌 You can also search any topics in the next page>>>
                """)
    
        























